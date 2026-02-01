// Copyright 2026 Alexander Orlov <alexander.orlov@loxal.net>

use std::path::Path;

type AnyError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Headers that are browser-internal (HTTP/2 pseudo-headers) or automatically
/// managed by HTTP clients — skip them when generating .hurl output.
const SKIP_REQUEST_HEADERS: &[&str] = &[
    ":authority",
    ":method",
    ":path",
    ":scheme",
    "accept-encoding",
    "cache-control",
    "pragma",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
    "sec-fetch-dest",
    "sec-fetch-mode",
    "sec-fetch-site",
    "sec-fetch-user",
    "sec-gpc",
    "dnt",
    "priority",
    "upgrade-insecure-requests",
];

// ---------------------------------------------------------------------------
// HAR → hurl
// ---------------------------------------------------------------------------

/// Convert a `.har` file to `.hurl` format and write the output to the given directory.
///
/// Returns the path of the generated `.hurl` file.
pub fn har_to_hurl(har_path: &Path, out_dir: &Path) -> Result<std::path::PathBuf, AnyError> {
    let har = har::from_path(har_path).map_err(|e| -> AnyError {
        format!("Failed to parse HAR file {}: {e}", har_path.display()).into()
    })?;

    let entries = match &har.log {
        har::Spec::V1_2(log) => &log.entries,
        har::Spec::V1_3(log) => {
            return har_v1_3_to_hurl(log, har_path, out_dir);
        }
    };

    let mut hurl_output = String::new();

    for (i, entry) in entries.iter().enumerate() {
        if i > 0 {
            hurl_output.push('\n');
        }

        let req = &entry.request;

        // Method + URL
        hurl_output.push_str(&format!("{} {}\n", req.method, req.url));

        // Request headers (filter out browser noise)
        for h in &req.headers {
            if SKIP_REQUEST_HEADERS.contains(&h.name.to_lowercase().as_str()) {
                continue;
            }
            hurl_output.push_str(&format!("{}: {}\n", h.name, h.value));
        }

        // Request cookies
        if !req.cookies.is_empty() {
            let cookie_str: String = req
                .cookies
                .iter()
                .map(|c| format!("{}={}", c.name, c.value))
                .collect::<Vec<_>>()
                .join("; ");
            hurl_output.push_str(&format!("cookie: {cookie_str}\n"));
        }

        // Request body (postData)
        if let Some(ref pd) = req.post_data
            && let Some(ref text) = pd.text
            && !text.is_empty()
        {
            if pd.mime_type.contains("json") {
                hurl_output.push_str(&format!("```json\n{text}\n```\n"));
            } else {
                hurl_output.push_str(&format!("```\n{text}\n```\n"));
            }
        }

        // Response section: status code + HTTP version
        let resp = &entry.response;
        let http_ver = normalize_http_version(&resp.http_version);
        hurl_output.push_str(&format!("{http_ver} {}\n", resp.status));
    }

    let out_name = har_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "converted".to_string());
    let out_path = out_dir.join(format!("{out_name}.hurl"));

    std::fs::write(&out_path, &hurl_output).map_err(|e| -> AnyError {
        format!("Failed to write {}: {e}", out_path.display()).into()
    })?;

    Ok(out_path)
}

/// HAR v1.3 → hurl (same logic, different types).
fn har_v1_3_to_hurl(
    log: &har::v1_3::Log,
    har_path: &Path,
    out_dir: &Path,
) -> Result<std::path::PathBuf, AnyError> {
    let mut hurl_output = String::new();

    for (i, entry) in log.entries.iter().enumerate() {
        if i > 0 {
            hurl_output.push('\n');
        }

        let req = &entry.request;
        hurl_output.push_str(&format!("{} {}\n", req.method, req.url));

        for h in &req.headers {
            if SKIP_REQUEST_HEADERS.contains(&h.name.to_lowercase().as_str()) {
                continue;
            }
            hurl_output.push_str(&format!("{}: {}\n", h.name, h.value));
        }

        if !req.cookies.is_empty() {
            let cookie_str: String = req
                .cookies
                .iter()
                .map(|c| format!("{}={}", c.name, c.value))
                .collect::<Vec<_>>()
                .join("; ");
            hurl_output.push_str(&format!("cookie: {cookie_str}\n"));
        }

        if let Some(ref pd) = req.post_data
            && let Some(ref text) = pd.text
            && !text.is_empty()
        {
            if pd.mime_type.contains("json") {
                hurl_output.push_str(&format!("```json\n{text}\n```\n"));
            } else {
                hurl_output.push_str(&format!("```\n{text}\n```\n"));
            }
        }

        let resp = &entry.response;
        let http_ver = normalize_http_version(&resp.http_version);
        hurl_output.push_str(&format!("{http_ver} {}\n", resp.status));
    }

    let out_name = har_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "converted".to_string());
    let out_path = out_dir.join(format!("{out_name}.hurl"));

    std::fs::write(&out_path, &hurl_output).map_err(|e| -> AnyError {
        format!("Failed to write {}: {e}", out_path.display()).into()
    })?;

    Ok(out_path)
}

// ---------------------------------------------------------------------------
// hurl → HAR
// ---------------------------------------------------------------------------

/// Convert a `.hurl` file to `.har` format and write the output to the given directory.
///
/// Returns the path of the generated `.har` file.
pub fn hurl_to_har(hurl_path: &Path, out_dir: &Path) -> Result<std::path::PathBuf, AnyError> {
    let content = std::fs::read_to_string(hurl_path).map_err(|e| -> AnyError {
        format!("Failed to read {}: {e}", hurl_path.display()).into()
    })?;

    let hurl_file = hurl_core::parser::parse_hurl_file(&content).map_err(|e| -> AnyError {
        format!("Failed to parse {}: {e:?}", hurl_path.display()).into()
    })?;

    let now = chrono_now_iso();
    let mut har_entries: Vec<har::v1_2::Entries> = Vec::new();

    for entry in &hurl_file.entries {
        let req = &entry.request;

        let method = req.method.to_string();
        let url = template_to_string(&req.url);

        // Query string params from [QueryStringParams]
        let query_params: Vec<har::v1_2::QueryString> = req
            .querystring_params()
            .iter()
            .map(|kv| har::v1_2::QueryString {
                name: template_to_string(&kv.key),
                value: template_to_string(&kv.value),
                comment: None,
            })
            .collect();

        // Append query string to URL if present
        let full_url = if query_params.is_empty() {
            url.clone()
        } else {
            let sep = if url.contains('?') { '&' } else { '?' };
            let qs = query_params
                .iter()
                .map(|q| format!("{}={}", q.name, q.value))
                .collect::<Vec<_>>()
                .join("&");
            format!("{url}{sep}{qs}")
        };

        // Headers
        let headers: Vec<har::v1_2::Headers> = req
            .headers
            .iter()
            .map(|kv| har::v1_2::Headers {
                name: template_to_string(&kv.key),
                value: template_to_string(&kv.value),
                comment: None,
            })
            .collect();

        // Cookies
        let cookies: Vec<har::v1_2::Cookies> = req
            .cookies()
            .iter()
            .map(|c| har::v1_2::Cookies {
                name: template_to_string(&c.name),
                value: template_to_string(&c.value),
                path: None,
                domain: None,
                expires: None,
                http_only: None,
                secure: None,
                comment: None,
            })
            .collect();

        // Body → postData
        let post_data = req.body.as_ref().and_then(|body| {
            extract_hurl_body_text(body).map(|text| {
                let mime_type = guess_mime_type(&text);
                har::v1_2::PostData {
                    mime_type,
                    text: Some(text),
                    params: None,
                    comment: None,
                }
            })
        });

        // Determine HTTP version from [Options]
        let http_version = extract_hurl_http_version(req.options());

        let body_size = post_data
            .as_ref()
            .and_then(|pd| pd.text.as_ref())
            .map(|t| t.len() as i64)
            .unwrap_or(0);

        let har_request = har::v1_2::Request {
            method,
            url: full_url,
            http_version: http_version.clone(),
            cookies,
            headers,
            query_string: query_params,
            post_data,
            headers_size: -1,
            body_size,
            comment: None,
        };

        // Build response from hurl response section (if present)
        let har_response = if let Some(ref resp) = entry.response {
            let status = match &resp.status.value {
                hurl_core::ast::StatusValue::Specific(code) => *code as i64,
                hurl_core::ast::StatusValue::Any => 0,
            };
            har::v1_2::Response {
                status,
                status_text: String::new(),
                http_version,
                cookies: vec![],
                headers: vec![],
                content: har::v1_2::Content::default(),
                redirect_url: None,
                headers_size: -1,
                body_size: -1,
                comment: None,
            }
        } else {
            har::v1_2::Response {
                status: 0,
                status_text: String::new(),
                http_version,
                cookies: vec![],
                headers: vec![],
                content: har::v1_2::Content::default(),
                redirect_url: None,
                headers_size: -1,
                body_size: -1,
                comment: None,
            }
        };

        har_entries.push(har::v1_2::Entries {
            pageref: None,
            started_date_time: now.clone(),
            time: 0.0,
            request: har_request,
            response: har_response,
            cache: har::v1_2::Cache::default(),
            timings: har::v1_2::Timings {
                blocked: Some(-1.0),
                dns: Some(-1.0),
                connect: Some(-1.0),
                send: 0.0,
                wait: 0.0,
                receive: 0.0,
                ssl: Some(-1.0),
                comment: None,
            },
            server_ip_address: None,
            connection: None,
            comment: None,
        });
    }

    let log = har::v1_2::Log {
        creator: har::v1_2::Creator {
            name: "hurl-to-har-to-hurl-converter".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            comment: None,
        },
        browser: None,
        pages: None,
        entries: har_entries,
        comment: None,
    };

    let har = har::Har {
        log: har::Spec::V1_2(log),
    };

    let json = har::to_json(&har).map_err(|e| -> AnyError {
        format!("Failed to serialize HAR: {e}").into()
    })?;

    let out_name = hurl_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "converted".to_string());
    let out_path = out_dir.join(format!("{out_name}.har"));

    std::fs::write(&out_path, &json).map_err(|e| -> AnyError {
        format!("Failed to write {}: {e}", out_path.display()).into()
    })?;

    Ok(out_path)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Render a hurl Template to a plain string (no variable substitution in convert mode).
fn template_to_string(t: &hurl_core::ast::Template) -> String {
    use hurl_core::ast::TemplateElement;
    let mut out = String::new();
    for elem in &t.elements {
        match elem {
            TemplateElement::String { value, .. } => out.push_str(value),
            TemplateElement::Placeholder(p) => {
                out.push_str(&format!("{{{{{}}}}}", p.expr.kind));
            }
        }
    }
    out
}

/// Extract body text from a hurl Body node.
fn extract_hurl_body_text(body: &hurl_core::ast::Body) -> Option<String> {
    use hurl_core::ast::Bytes;
    use hurl_core::types::ToSource;
    match &body.value {
        Bytes::Json(jv) => Some(jv.to_source().as_str().to_string()),
        Bytes::Xml(s) => Some(s.clone()),
        Bytes::MultilineString(m) => Some(m.value().to_string()),
        Bytes::OnelineString(t) => Some(template_to_string(t)),
        Bytes::Base64(b) => String::from_utf8(b.value.clone()).ok(),
        Bytes::Hex(h) => String::from_utf8(h.value.clone()).ok(),
        Bytes::File(_) => None, // Cannot inline file content without reading it
    }
}

/// Guess MIME type from body text content.
fn guess_mime_type(text: &str) -> String {
    let trimmed = text.trim();
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        "application/json".to_string()
    } else if trimmed.starts_with('<') {
        "application/xml".to_string()
    } else {
        "text/plain".to_string()
    }
}

/// Extract HTTP version string from hurl [Options].
fn extract_hurl_http_version(options: &[hurl_core::ast::EntryOption]) -> String {
    use hurl_core::ast::{BooleanOption, OptionKind};
    for opt in options {
        match &opt.kind {
            OptionKind::Http10(BooleanOption::Literal(true)) => return "HTTP/1.0".to_string(),
            OptionKind::Http11(BooleanOption::Literal(true)) => return "HTTP/1.1".to_string(),
            OptionKind::Http2(BooleanOption::Literal(true)) => return "HTTP/2".to_string(),
            OptionKind::Http3(BooleanOption::Literal(true)) => return "HTTP/3".to_string(),
            _ => {}
        }
    }
    "HTTP/1.1".to_string()
}

/// Normalize HAR httpVersion strings to hurl-compatible format.
///
/// HAR uses "http/2.0", "http/1.1", "h2" etc. Hurl expects "HTTP/2", "HTTP/1.1" etc.
fn normalize_http_version(version: &str) -> String {
    match version.to_lowercase().as_str() {
        "http/2.0" | "http/2" | "h2" | "h2c" => "HTTP/2".to_string(),
        "http/3.0" | "http/3" | "h3" => "HTTP/3".to_string(),
        "http/1.0" | "h10" | "http/0.9" | "h09" => "HTTP/1.0".to_string(),
        "http/1.1" | "h1" => "HTTP/1.1".to_string(),
        _ => "HTTP/1.1".to_string(),
    }
}

/// Simple ISO 8601 timestamp using system time (no chrono dependency).
fn chrono_now_iso() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}.000Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days: &[u64] = if is_leap(year) {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

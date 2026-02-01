# hurl-to-har-to-hurl-converter

Convert between [HURL](https://hurl.dev) and [HAR](http://www.softwareishard.com/blog/har-12-spec/) file formats.

## Build

```bash
cargo build --release -p hurl-to-har-to-hurl-converter
```

The binary will be at `target/release/hurl-to-har-to-hurl-converter`.

## Usage

The tool auto-detects the conversion direction from the input file extension.

### HURL to HAR

```bash
hurl-to-har-to-hurl-converter request.hurl
```

Produces `request.har` in the current directory.

### HAR to HURL

```bash
hurl-to-har-to-hurl-converter export.har
```

Produces `export.hurl` in the current directory.

### Custom output directory

```bash
hurl-to-har-to-hurl-converter request.hurl --output-dir /tmp/converted
```

The output directory must already exist.

## Conversion details

### HAR to HURL

- Supports HAR v1.2 and v1.3
- Filters browser-internal headers (HTTP/2 pseudo-headers, `sec-fetch-*`, `sec-ch-ua-*`, etc.)
- Combines request cookies into a single `cookie:` header
- Includes request body as a fenced code block (` ```json ` for JSON content)
- Maps response status code and HTTP version

### HURL to HAR

- Parses HURL files using `hurl_core`
- Extracts method, URL, headers, cookies, query string params, and body
- Preserves `{{variable}}` placeholders as literal text
- Detects HTTP version from `[Options]` section (`http10`, `http11`, `http2`, `http3`)
- Auto-detects body MIME type (JSON, XML, or plain text)
- Outputs HAR v1.2 JSON

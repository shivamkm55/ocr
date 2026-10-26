import { useEffect, useRef, useState } from "react";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

function Icon({ name, size = 20 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };

  const paths = {
    sparkles: (
      <>
        <path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-2-5.8L4 11l6-2.2L12 3Z" />
        <path d="m19 14 .9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9L19 14Z" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
        <path d="M20 16.5v2A1.5 1.5 0 0 1 18.5 20h-13A1.5 1.5 0 0 1 4 18.5v-2" />
      </>
    ),
    image: (
      <>
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <path d="m21 15-5-5L5 21" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="13" height="13" rx="2" />
        <path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" />
      </>
    ),
    download: (
      <>
        <path d="M12 4v11m0 0 4-4m-4 4-4-4" />
        <path d="M5 17v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" />
      </>
    ),
    close: <path d="m18 6-12 12M6 6l12 12" />,
    check: <path d="m5 12 4 4L19 6" />,
    file: (
      <>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
        <path d="M14 2v6h6M8 13h8m-8 4h8" />
      </>
    ),
    arrow: <path d="M5 12h14m-7-7 7 7-7 7" />,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

function formatSize(bytes) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(0)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function reset() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setCopied(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  function selectFile(selected) {
    if (!selected) return;
    setError("");
    setResult(null);
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setError("Please choose a JPG, PNG, or WebP image.");
      return;
    }
    if (selected.size > MAX_FILE_SIZE) {
      setError("This image is larger than 10 MB. Choose a smaller file to continue.");
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  async function extractText() {
    if (!file || loading) return;
    setLoading(true);
    setError("");
    setResult(null);

    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch("/ocr", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Text extraction failed. Please try again.");
      }
      setResult(payload);
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? "Could not reach the OCR service. Check that the server is running and try again."
          : requestError.message,
      );
    } finally {
      setLoading(false);
    }
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setError("Clipboard access was blocked. Select the text and copy it manually.");
    }
  }

  function downloadText() {
    const blob = new Blob([result.text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${file.name.replace(/\.[^.]+$/, "") || "extracted-text"}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Typeframe home">
          <span className="brand-mark"><Icon name="sparkles" size={19} /></span>
          <span>typeframe<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-right">
          <span className="engine-status"><span className="status-dot" /> Google Vision AI</span>
          <a className="github-link" href="#how-it-works">How it works <Icon name="arrow" size={15} /></a>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow"><span className="eyebrow-line" /> IMAGE TO TEXT, MADE SIMPLE</div>
          <h1>Your words are<br />already <span>in there.</span></h1>
          <p className="hero-copy">Pull text from photos, scans, and screenshots in seconds.<br className="desktop-break" /> Just drop an image and let it happen.</p>
        </section>

        <section className="workspace" aria-label="Image text extraction">
          <div className="workspace-heading">
            <div>
              <p className="section-kicker">THE WORKSPACE</p>
              <h2>Extract your text</h2>
            </div>
            <span className="privacy-note"><span className="privacy-dot" /> Sent to Google Vision for processing</span>
          </div>

          <div
            className={`dropzone ${dragging ? "dropzone-active" : ""} ${file ? "dropzone-filled" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              selectFile(event.dataTransfer.files[0]);
            }}
          >
            {file ? (
              <div className="file-selected">
                <div className="file-preview">
                  <img src={preview} alt={`Preview of ${file.name}`} />
                </div>
                <div className="file-details">
                  <span className="file-label">READY TO EXTRACT</span>
                  <strong title={file.name}>{file.name}</strong>
                  <span className="file-size">{formatSize(file.size)} <span>·</span> {file.type.split("/")[1].toUpperCase()}</span>
                </div>
                <button className="icon-button remove-file" onClick={reset} aria-label="Remove image" disabled={loading}>
                  <Icon name="close" size={17} />
                </button>
              </div>
            ) : (
              <div className="drop-content">
                <div className="upload-icon"><Icon name="upload" size={22} /></div>
                <div className="drop-copy">
                  <strong>Drop your image here</strong>
                  <span>or <button className="browse-button" onClick={() => inputRef.current?.click()}>browse files</button> from your device</span>
                </div>
                <span className="file-types">JPG <i /> PNG <i /> WEBP <i /> UP TO 10 MB</span>
              </div>
            )}
            <input
              ref={inputRef}
              className="file-input"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => selectFile(event.target.files[0])}
              aria-label="Choose an image"
            />
          </div>

          {error && <div className="error-message" role="alert">{error}</div>}

          <button className="extract-button" onClick={extractText} disabled={!file || loading}>
            {loading ? (
              <><span className="spinner" /> Reading your image...</>
            ) : result ? (
              <><Icon name="check" size={18} /> Text extracted <span className="button-arrow">✓</span></>
            ) : (
              <>Extract text <Icon name="arrow" size={17} /></>
            )}
          </button>

          {result && (
            <section className="result-card" aria-live="polite">
              <div className="result-header">
                <div>
                  <span className="section-kicker">THE RESULT</span>
                  <h3>Your extracted text</h3>
                </div>
                <div className="result-actions">
                  <button onClick={copyText} className={`secondary-button ${copied ? "button-success" : ""}`}>
                    <Icon name={copied ? "check" : "copy"} size={15} /> {copied ? "Copied" : "Copy"}
                  </button>
                  <button onClick={downloadText} className="secondary-button">
                    <Icon name="download" size={15} /> Save .txt
                  </button>
                </div>
              </div>
              <textarea
                className="result-text"
                value={result.text}
                onChange={(event) => setResult({ ...result, text: event.target.value })}
                aria-label="Extracted text, editable"
                placeholder="No text was detected in this image."
                spellCheck={false}
              />
              <div className="result-footer">
                <span><Icon name="file" size={14} /> {file.name}</span>
                <span>{result.text.length.toLocaleString()} characters</span>
              </div>
            </section>
          )}
        </section>

        <section className="how-section" id="how-it-works">
          <div className="how-heading">
            <p className="section-kicker">THREE STEPS. THAT'S IT.</p>
            <h2>From image to insight.</h2>
          </div>
          <div className="steps">
            <article className="step">
              <span className="step-number">01</span>
              <div className="step-icon"><Icon name="image" size={20} /></div>
              <h3>Drop it in</h3>
              <p>Upload a photo, scan, or screenshot. JPG, PNG, and WebP all work.</p>
            </article>
            <article className="step">
              <span className="step-number">02</span>
              <div className="step-icon step-icon-accent"><Icon name="sparkles" size={20} /></div>
              <h3>We read it</h3>
              <p>Google Vision AI finds and recognizes text, even in complex layouts.</p>
            </article>
            <article className="step">
              <span className="step-number">03</span>
              <div className="step-icon"><Icon name="copy" size={20} /></div>
              <h3>Make it yours</h3>
              <p>Edit, copy, or save the extracted text. It's ready when you are.</p>
            </article>
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand footer-brand" href="/"><span className="brand-mark"><Icon name="sparkles" size={16} /></span><span>typeframe<span className="brand-period">.</span></span></a>
        <span>Less retyping. More doing.</span>
        <span className="footer-powered">POWERED BY GOOGLE CLOUD VISION</span>
      </footer>
    </div>
  );
}

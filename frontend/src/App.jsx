import { useEffect, useRef, useState } from "react";
import VisualContext, { buildAIReadyText, buildExport } from "./VisualContext";

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
  const [includeContext, setIncludeContext] = useState(true);
  const [contextMode, setContextMode] = useState("workflow");
  const [activeTab, setActiveTab] = useState("text");
  const copyTimerRef = useRef(null);
  const tabsRef = useRef(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  useEffect(() => () => window.clearTimeout(copyTimerRef.current), []);

  function reset() {
    if (loading) return;
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setCopied(false);
    setActiveTab("text");
    if (inputRef.current) inputRef.current.value = "";
  }

  function selectFile(selected) {
    if (!selected || loading) return;
    setError("");
    setResult(null);
    setCopied(false);
    setActiveTab("text");
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
    body.append("include_context", String(includeContext));
    body.append("context_mode", contextMode);

    try {
      const response = await fetch("/ocr", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Text extraction failed. Please try again.");
      }
      setResult({ ...payload, text: payload.text || "", original_text: payload.text || "" });
      setActiveTab(payload.visual_context ? "context" : "text");
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

  async function copyText(value = result.text, kind = "text") {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(kind);
      window.clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setError("Clipboard access was blocked. Select the text and copy it manually.");
    }
  }

  function downloadText(value = result.text, extension = "txt") {
    const blob = new Blob([value], { type: extension === "json" ? "application/json;charset=utf-8" : "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${file.name.replace(/\.[^.]+$/, "") || "extracted-text"}.${extension}`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function handleTabKey(event) {
    const tabs = [...tabsRef.current.querySelectorAll('[role="tab"]')];
    const index = tabs.indexOf(event.target);
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      tabs[next].focus();
      setActiveTab(tabs[next].dataset.tab);
    }
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
          <div className="eyebrow"><span className="eyebrow-line" /> TEXT, LAYOUT, AND THE CONNECTIONS BETWEEN</div>
          <h1>Your image has<br /><span>more to say.</span></h1>
          <p className="hero-copy">Extract the words and the visual context around them.<br className="desktop-break" /> Built for workflows, diagrams, and code screenshots.</p>
        </section>

        <section className="workspace" aria-label="Image text and visual context extraction">
          <div className="workspace-heading">
            <div>
              <p className="section-kicker">THE WORKSPACE</p>
              <h2>Read your image</h2>
            </div>
            <span className="privacy-note"><span className="privacy-dot" /> Sent to Google Vision for processing</span>
          </div>

          <div
            className={`dropzone ${dragging ? "dropzone-active" : ""} ${file ? "dropzone-filled" : ""}`}
            onDragOver={(event) => { event.preventDefault(); if (!loading) setDragging(true); }}
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
              disabled={loading}
              onChange={(event) => selectFile(event.target.files[0])}
              aria-label="Choose an image"
            />
          </div>

          <div className="analysis-options">
            <label className="context-option">
              <input type="checkbox" checked={includeContext} onChange={(event) => setIncludeContext(event.target.checked)} disabled={loading} />
              <span><strong>Include visual context</strong><small>Layout, colours, grouping, and candidate connections</small></span>
            </label>
            <fieldset className="analysis-mode" disabled={!includeContext || loading}>
              <legend>Image type</legend>
              <label className={contextMode === "workflow" ? "mode-selected" : ""}><input type="radio" name="context-mode" value="workflow" checked={contextMode === "workflow"} onChange={() => setContextMode("workflow")} />Workflow</label>
              <label className={contextMode === "code" ? "mode-selected" : ""}><input type="radio" name="context-mode" value="code" checked={contextMode === "code"} onChange={() => setContextMode("code")} />Code screenshot</label>
            </fieldset>
          </div>

          {error && <div className="error-message" role="alert">{error}</div>}

          <button className="extract-button" onClick={extractText} disabled={!file || loading}>
            {loading ? (
              <><span className="spinner" /> {includeContext ? "Reading text and visual context..." : "Reading your image..."}</>
            ) : result ? (
              <>Read image again <Icon name="arrow" size={17} /></>
            ) : (
              <>{includeContext ? "Extract text + context" : "Extract text"} <Icon name="arrow" size={17} /></>
            )}
          </button>

          {result && (
            <section className="result-card" aria-live="polite">
              <div className="result-header">
                <div>
                  <span className="section-kicker">THE RESULT</span>
                  <h3>{result.visual_context ? "The words. The bigger picture." : "Your extracted text"}</h3>
                </div>
                <div className="result-actions">
                  <button onClick={() => copyText(buildAIReadyText(result), "bundle")} className={`secondary-button ${copied === "bundle" ? "button-success" : ""}`}>
                    <Icon name={copied === "bundle" ? "check" : "copy"} size={15} /> {copied === "bundle" ? "Copied" : "Copy for AI"}
                  </button>
                </div>
              </div>
              {result.context_error && <div className="context-warning" role="status"><strong>Your text is ready.</strong> Visual context could not be completed: {result.context_error}</div>}
              <div className="result-tabs" role="tablist" aria-label="Extraction results" ref={tabsRef} onKeyDown={handleTabKey}>
                {[{ id: "text", label: "Text" }, { id: "context", label: "Visual context" }, { id: "data", label: "Structured data" }].map((tab) => <button key={tab.id} id={`tab-${tab.id}`} role="tab" data-tab={tab.id} aria-selected={activeTab === tab.id} aria-controls={`panel-${tab.id}`} tabIndex={activeTab === tab.id ? 0 : -1} onClick={() => setActiveTab(tab.id)}>{tab.label}{tab.id === "context" && result.visual_context && <span className="tab-dot" />}</button>)}
              </div>
              <div role="tabpanel" id="panel-text" aria-labelledby="tab-text" hidden={activeTab !== "text"} tabIndex="0">
                <div className="panel-heading"><p>Edit any words before copying.</p><div className="result-actions"><button onClick={() => copyText()} className={`secondary-button ${copied === "text" ? "button-success" : ""}`}><Icon name={copied === "text" ? "check" : "copy"} size={14} />{copied === "text" ? "Copied" : "Copy text"}</button><button onClick={() => downloadText()} className="secondary-button"><Icon name="download" size={14} />Save .txt</button></div></div>
                <textarea className="result-text" value={result.text} onChange={(event) => setResult({ ...result, text: event.target.value })} aria-label="Extracted text, editable" placeholder="No text was detected in this image." spellCheck={false} />
                {result.text !== result.original_text && result.visual_context && <p className="context-caption">Your edits are included in copies and downloads. Region labels still reference the original image text.</p>}
              </div>
              <div role="tabpanel" id="panel-context" aria-labelledby="tab-context" hidden={activeTab !== "context"} tabIndex="0">
                {result.visual_context ? <VisualContext context={result.visual_context} preview={preview} filename={result.filename} /> : <div className="context-unavailable"><Icon name="image" size={26} /><h4>{result.context_error ? "Visual context is unavailable" : "Only text was extracted"}</h4><p>{result.context_error ? "Your extracted text is still available in the Text tab. You can retry the image to request visual context again." : "Enable Include visual context above and read the image again to capture its layout, colours, and connections."}</p></div>}
              </div>
              <div role="tabpanel" id="panel-data" aria-labelledby="tab-data" hidden={activeTab !== "data"} tabIndex="0">
                <div className="panel-heading"><p>Text and observations in one portable file.</p><button onClick={() => downloadText(JSON.stringify(buildExport(result), null, 2), "json")} className="secondary-button"><Icon name="download" size={14} />Save .json</button></div>
                <pre className="structured-data" tabIndex="0" aria-label="Structured extraction data">{JSON.stringify(buildExport(result), null, 2)}</pre>
              </div>
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
              <p>Upload a workflow, diagram, or code screenshot. JPG, PNG, and WebP all work.</p>
            </article>
            <article className="step">
              <span className="step-number">02</span>
              <div className="step-icon step-icon-accent"><Icon name="sparkles" size={20} /></div>
              <h3>Read the full picture</h3>
              <p>Extract text with Google Vision, then inspect layout, colours, and possible connections.</p>
            </article>
            <article className="step">
              <span className="step-number">03</span>
              <div className="step-icon"><Icon name="copy" size={20} /></div>
              <h3>Make it yours</h3>
              <p>Review the observations, edit the text, and copy both into your next AI conversation.</p>
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

/**
 * Render an HTML report to PDF with real page numbers and footers.
 *
 * Drives the Chromium-based browser already installed on the machine
 * (Brave/Chrome/Edge) through puppeteer-core, because Chrome's bundled CLI
 * `--print-to-pdf` cannot inject a dynamic "Page N of M" footer.
 *
 * Usage:
 *   node render_pdf.js <input.html> <output.pdf> [title] [footerNote]
 */

const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

const CANDIDATES = [
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
];

function findBrowser() {
  for (const c of CANDIDATES) {
    try {
      fs.accessSync(c, fs.constants.X_OK);
      return c;
    } catch (_) {
      /* keep looking */
    }
  }
  throw new Error(
    "No Chromium-based browser found. Tried:\n  " + CANDIDATES.join("\n  ")
  );
}

async function main() {
  const [, , input, output, title = "", footerNote = ""] = process.argv;
  if (!input || !output) {
    console.error("usage: node render_pdf.js <input.html> <output.pdf> [title] [note]");
    process.exit(2);
  }

  const browserPath = findBrowser();
  const htmlPath = path.resolve(input);

  const browser = await puppeteer.launch({
    executablePath: browserPath,
    headless: "new",
    args: ["--no-sandbox", "--disable-gpu", "--font-render-hinting=none"],
  });

  try {
    const page = await browser.newPage();
    await page.goto("file://" + htmlPath, { waitUntil: "networkidle0", timeout: 180000 });

    // Give inline SVG layout a chance to settle before snapshotting.
    await page.evaluateHandle("document.fonts.ready");
    await new Promise((r) => setTimeout(r, 900));

    const footer = `
      <div style="width:100%;padding:0 14mm;font-family:Helvetica,Arial,sans-serif;
                  font-size:8px;color:#5A6B7B;display:flex;
                  justify-content:space-between;align-items:center;">
        <span>${footerNote}</span>
        <span>${title}</span>
        <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
      </div>`;

    await page.pdf({
      path: output,
      format: "A4",
      printBackground: true,
      displayHeaderFooter: true,
      headerTemplate: "<div></div>",
      footerTemplate: footer,
      margin: { top: "14mm", right: "14mm", bottom: "16mm", left: "14mm" },
      timeout: 180000,
    });

    // Report the real page count back to the caller.
    const buf = fs.readFileSync(output);
    const matches = buf.toString("latin1").match(/\/Type\s*\/Page[^s]/g);
    const pages = matches ? matches.length : 0;
    console.log(`  rendered ${output} (${(buf.length / 1e6).toFixed(2)} MB, ${pages} pages)`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error("PDF render failed:", err.message);
  process.exit(1);
});

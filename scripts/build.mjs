/**
 * Builds the site. Uses only Node's standard library — no npm packages,
 * no node_modules, nothing to keep up to date.
 *
 *   node scripts/build.mjs            build published posts
 *   node scripts/build.mjs --drafts   include posts marked draft: true
 *
 * Reads:  posts/*.md, data/profile.json, data/github.json, templates/base.html
 * Writes: index.html, writing/, about/, 404.html, feed.xml
 */

import { readFile, writeFile, readdir, mkdir, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const p = (...s) => path.join(ROOT, ...s);
const INCLUDE_DRAFTS = process.argv.includes("--drafts");

/* ============================ markdown ============================ */

const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Inline formatting. Input is already HTML-escaped. */
function inline(text) {
  return text
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) => `<img src="${src}" alt="${alt}" loading="lazy">`)
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => {
      const external = /^https?:\/\//.test(href);
      return `<a href="${href}"${external ? ' rel="noopener"' : ""}>${label}</a>`;
    })
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(/ -- /g, " &mdash; ");
}

/* ---------- tables (GFM pipe syntax) ---------- */

/** The second row of a table: `| :--- | :---: | ---: |`. It must contain a pipe. */
const TABLE_DELIMITER = /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?$/;

function isTable(lines) {
  return lines.length >= 2 && lines.every((l) => l.includes("|")) && TABLE_DELIMITER.test(lines[1].trim());
}

/** `| a | b \| c |` → ["a", "b | c"]. A literal pipe inside a cell is written \|. */
function splitRow(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/(?<!\\)\|$/, "")
    .split(/(?<!\\)\|/)
    .map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

function alignmentOf(delimiter) {
  const left = delimiter.startsWith(":");
  const right = delimiter.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  if (left) return "left";
  return "";
}

/**
 * Header row, delimiter row, body rows. Alignment comes only from the
 * delimiter row: write ---: for a numeric column, and the stylesheet
 * right-aligns it with tabular figures. The table sits in a scroll
 * container so a wide one never widens the page.
 */
function table(lines) {
  const header = splitRow(lines[0]);
  const aligns = splitRow(lines[1]).map(alignmentOf);
  // GFM rule: every body row has exactly as many cells as the header.
  const rows = lines.slice(2).map((line) => {
    const cells = splitRow(line);
    return header.map((_, i) => cells[i] ?? "");
  });

  const classes = aligns.map((align) => (align === "center" || align === "right" ? `align-${align}` : ""));

  const cell = (tag, text, i) => {
    const cls = classes[i] ? ` class="${classes[i]}"` : "";
    return `<${tag}${cls}>${inline(escapeHtml(text))}</${tag}>`;
  };
  const head = `<tr>${header.map((text, i) => cell("th", text, i)).join("")}</tr>`;
  const body = rows.map((row) => `<tr>${row.map((text, i) => cell("td", text, i)).join("")}</tr>`).join("\n");

  return `<div class="table-wrap" tabindex="0"><table>\n<thead>${head}</thead>\n<tbody>\n${body}\n</tbody>\n</table></div>`;
}

/**
 * A deliberately small Markdown subset: headings (##, ###), paragraphs,
 * bullet and numbered lists, blockquotes, fenced code, rules, images,
 * GFM pipe tables, bold, italic, inline code and links. Raw HTML blocks
 * pass through. Lists do not nest — that limit keeps this readable and
 * predictable.
 */
function markdown(src) {
  const fences = [];
  let text = src.replace(/\r\n?/g, "\n");
  // A "blank" line that carries invisible trailing whitespace (common after
  // copy-pasting from a browser or word processor) would otherwise defeat the
  // \n{2,} paragraph split below, silently merging separate paragraphs into
  // one. Normalize any whitespace-only line to truly empty first.
  text = text.replace(/^[ \t]+$/gm, "");

  // Pull fenced code out first so nothing else touches it.
  text = text.replace(/```([\w-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const cls = lang ? ` class="language-${lang}"` : "";
    fences.push(`<pre><code${cls}>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000FENCE${fences.length - 1}\u0000`;
  });

  const html = text
    .split(/\n{2,}/)
    .map((raw) => {
      const block = raw.trim();
      if (!block) return "";

      if (/^\u0000FENCE\d+\u0000$/.test(block)) return block;
      if (/^(---|\*\*\*)$/.test(block)) return "<hr>";
      if (block.startsWith("<")) return block; // raw HTML escape hatch

      const heading = block.match(/^(#{2,4})\s+(.*)$/s);
      if (heading) {
        const level = heading[1].length;
        return `<h${level}>${inline(escapeHtml(heading[2].trim()))}</h${level}>`;
      }

      const lines = block.split("\n");

      if (isTable(lines)) return table(lines);

      if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
        const items = lines.map((l) => `<li>${inline(escapeHtml(l.replace(/^\s*[-*]\s+/, "")))}</li>`);
        return `<ul>${items.join("")}</ul>`;
      }

      if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
        const items = lines.map((l) => `<li>${inline(escapeHtml(l.replace(/^\s*\d+[.)]\s+/, "")))}</li>`);
        return `<ol>${items.join("")}</ol>`;
      }

      if (lines.every((l) => /^\s*>\s?/.test(l))) {
        const inner = lines.map((l) => l.replace(/^\s*>\s?/, "")).join(" ");
        return `<blockquote><p>${inline(escapeHtml(inner))}</p></blockquote>`;
      }

      return `<p>${inline(escapeHtml(lines.join("\n")))}</p>`;
    })
    .filter(Boolean)
    .join("\n");

  return html.replace(/\u0000FENCE(\d+)\u0000/g, (_, i) => fences[Number(i)]);
}

/* ============================ posts ============================ */

function parseFrontMatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) return { meta: {}, body: raw };
  const meta = {};
  for (const line of match[1].split("\n")) {
    const kv = line.match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
    if (!kv) continue;
    let value = kv[2].trim().replace(/^["'](.*)["']$/, "$1");
    if (value === "true") value = true;
    else if (value === "false") value = false;
    meta[kv[1]] = value;
  }
  return { meta, body: raw.slice(match[0].length) };
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function humanDate(iso) {
  const [y, m, d] = String(iso).split("-").map(Number);
  if (!y || !m || !d) return String(iso);
  return `${d} ${MONTHS[m - 1]} ${y}`;
}

async function loadPosts() {
  let files = [];
  try {
    files = (await readdir(p("posts"))).filter((f) => f.endsWith(".md"));
  } catch {
    return [];
  }

  const posts = [];
  for (const file of files) {
    const raw = await readFile(p("posts", file), "utf8");
    const { meta, body } = parseFrontMatter(raw);
    if (meta.draft && !INCLUDE_DRAFTS) continue;
    if (!meta.title) {
      console.warn(`skipped posts/${file}: no title in front matter`);
      continue;
    }
    const slug = meta.slug || file.replace(/\.md$/, "");
    posts.push({
      slug,
      title: meta.title,
      date: meta.date || "1970-01-01",
      summary: meta.summary || "",
      linkedin: meta.linkedin || "",
      draft: Boolean(meta.draft),
      url: `writing/${slug}/`,
      html: markdown(body),
      words: body.split(/\s+/).filter(Boolean).length,
    });
  }
  return posts.sort((a, b) => String(b.date).localeCompare(String(a.date)));
}

/* ============================ rendering ============================ */

const site = JSON.parse(await readFile(p("data", "profile.json"), "utf8"));
const base = await readFile(p("templates", "base.html"), "utf8");

let gh = { repos: [], pulls: [], stats: {} };
try {
  gh = JSON.parse(await readFile(p("data", "github.json"), "utf8"));
} catch {
  console.warn("data/github.json missing — run node scripts/fetch-github.mjs");
}

const ROOT_PATH = site.baseurl || "/";
const SITE_URL = String(site.siteurl || "").replace(/\/$/, "");
const linkOf = (rel) => `${SITE_URL}/${rel}`.replace(/([^:])\/{2,}/g, "$1/");

function page({ title, description, body, canonical, ogtype = "website", current = "" }) {
  return base
    .replace(/\{\{lang\}\}/g, site.lang || "en")
    .replace(/\{\{title\}\}/g, escapeHtml(title))
    .replace(/\{\{ogtitle\}\}/g, escapeHtml(title))
    .replace(/\{\{description\}\}/g, escapeHtml(description))
    .replace(/\{\{canonical\}\}/g, canonical)
    .replace(/\{\{ogtype\}\}/g, ogtype)
    .replace(/\{\{ogimage\}\}/g, linkOf("assets/og-image.png"))
    .replace(/\{\{sitename\}\}/g, escapeHtml(site.name))
    .replace(/\{\{name\}\}/g, escapeHtml(site.name))
    .replace(/\{\{root\}\}/g, ROOT_PATH)
    .replace(/\{\{github\}\}/g, site.github)
    .replace(/\{\{linkedin\}\}/g, site.linkedin)
    .replace(/\{\{sourceurl\}\}/g, site.sourceurl || site.github)
    .replace(/\{\{year\}\}/g, String(new Date().getFullYear()))
    .replace(/\{\{navwriting\}\}/g, current === "writing" ? ' aria-current="page"' : "")
    .replace(/\{\{navabout\}\}/g, current === "about" ? ' aria-current="page"' : "")
    .replace(/\{\{body\}\}/g, body);
}

/** `2026 · 08` — the year-month prefix used in the writing list. */
const yearMonth = (iso) => `${String(iso).slice(0, 4)} · ${String(iso).slice(5, 7)}`;

function entryList(posts) {
  if (!posts.length) {
    return `    <p class="empty">No posts yet. Drop a Markdown file in <code>posts/</code> and push — the build does the rest.</p>`;
  }
  const items = posts
    .map(
      (post) => `      <li><a href="${ROOT_PATH}${post.url}"><span class="when">${yearMonth(post.date)}</span><span>${escapeHtml(post.title)}${post.draft ? " (draft)" : ""}</span></a></li>`,
    )
    .join("\n");
  return `    <ul class="post-list">\n${items}\n    </ul>`;
}

function repoRows(repos) {
  if (!repos.length) return "";
  const rows = repos
    .map((r) => {
      const desc = r.description || r.language || "";
      const stars = r.stars ? `<span class="sep">·</span>${r.stars} ★` : "";
      return `      <li><a class="lead" href="${r.url}" rel="noopener">${escapeHtml(r.name)}</a><span class="sep">—</span>${escapeHtml(desc)}${stars}</li>`;
    })
    .join("\n");
  return `    <ul class="plain-list">\n${rows}\n    </ul>`;
}

/** Upstream patches grouped by project — a count reads stronger than a list of similar titles. */
function upstreamRows(pulls) {
  const groups = new Map();
  for (const pull of pulls.filter((x) => x.upstream)) {
    const g = groups.get(pull.repo) || { repo: pull.repo, count: 0, latest: pull.created_at };
    g.count += 1;
    if (pull.created_at > g.latest) g.latest = pull.created_at;
    groups.set(pull.repo, g);
  }
  const list = [...groups.values()].sort((a, b) => b.count - a.count || b.latest.localeCompare(a.latest));
  if (!list.length) return "";

  const rows = list
    .map((g) => {
      const q = `https://github.com/${g.repo}/pulls?q=is%3Apr+author%3A${site.handle}`;
      return `      <li><a class="lead" href="${q}" rel="noopener">${escapeHtml(g.repo)}</a><span class="sep">—</span>${g.count} ${g.count === 1 ? "patch" : "patches"}</li>`;
    })
    .join("\n");
  return `    <ul class="plain-list">\n${rows}\n    </ul>`;
}

/* ---------- pages ---------- */

function homePage(posts) {
  const repos = (gh.repos || []).filter((r) => !(site.hidden || []).includes(r.name));

  const body = `    <p class="kicker"><a href="${ROOT_PATH}writing/">Writing</a></p>
${entryList(posts)}

    <hr class="divider">

    <p class="kicker"><a href="${site.github}?tab=repositories" rel="me">GitHub Repo</a></p>
${repoRows(repos)}

    <hr class="divider">

    <p class="kicker">Pull Requests</p>
${upstreamRows(gh.pulls || [])}`;

  return page({
    title: `${site.name} — ${site.role}`,
    description: site.description,
    canonical: `${SITE_URL}/`,
    ogtype: "profile",
    body,
  });
}

function writingIndexPage(posts) {
  const body = `    <h1 class="page-title">Writing</h1>
${entryList(posts)}`;
  return page({
    title: `Writing — ${site.name}`,
    description: `Paper notes and essays by ${site.name} on ${site.role.toLowerCase()}.`,
    canonical: linkOf("writing/"),
    current: "writing",
    body,
  });
}

function postPage(post) {
  const meta = [
    `<time datetime="${post.date}">${humanDate(post.date)}</time>`,
    `${Math.max(1, Math.round(post.words / 220))} minute read`,
  ];
  if (post.linkedin) meta.push(`<a href="${post.linkedin}" rel="noopener">Discussion on LinkedIn</a>`);

  const body = `    <article class="article">
      <h1>${escapeHtml(post.title)}</h1>
      <p class="meta">${meta.join(" · ")}</p>
      <div class="prose">
${post.html}
      </div>
      <p class="backlink"><a href="${ROOT_PATH}">&larr; Home</a></p>
    </article>`;

  return page({
    title: `${post.title} — ${site.name}`,
    description: post.summary || `${post.title} by ${site.name}.`,
    canonical: linkOf(post.url),
    ogtype: "article",
    current: "writing",
    body,
  });
}

function aboutPage() {
  const links = (site.links || [])
    .map((l) => `      <li><a class="lead" href="${l.url}" rel="me noopener">${escapeHtml(l.label)}</a><span class="sep">—</span>${escapeHtml(l.note || "")}</li>`)
    .join("\n");

  const body = `    <article class="article">
      <h1>About</h1>
      <div class="prose">
        ${(site.about || []).map((t) => `<p>${inline(escapeHtml(t))}</p>`).join("\n        ")}
      </div>
    </article>

    <hr class="divider">

    <p class="kicker">Elsewhere</p>
    <ul class="plain-list">
${links}
    </ul>`;

  return page({
    title: `About — ${site.name}`,
    description: site.description,
    canonical: linkOf("about/"),
    current: "about",
    body,
  });
}

function notFoundPage() {
  const body = `    <article class="article">
      <h1>That page isn't here.</h1>
      <div class="prose">
        <p>The link may be out of date. <a href="${ROOT_PATH}writing/">Browse the writing</a> or start from the <a href="${ROOT_PATH}">homepage</a>.</p>
      </div>
    </article>`;
  return page({ title: `Not found — ${site.name}`, description: "Page not found.", canonical: linkOf("404.html"), body });
}

function feed(posts) {
  const items = posts
    .slice(0, 20)
    .map(
      (post) => `    <item>
      <title>${escapeHtml(post.title)}</title>
      <link>${linkOf(post.url)}</link>
      <guid isPermaLink="true">${linkOf(post.url)}</guid>
      <pubDate>${new Date(`${post.date}T09:00:00Z`).toUTCString()}</pubDate>
      <description>${escapeHtml(post.summary || post.title)}</description>
    </item>`,
    )
    .join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${escapeHtml(site.name)}</title>
    <link>${SITE_URL}/</link>
    <atom:link href="${linkOf("feed.xml")}" rel="self" type="application/rss+xml"/>
    <description>${escapeHtml(site.description)}</description>
    <language>${site.lang || "en"}</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
${items}
  </channel>
</rss>
`;
}

/* ============================ write it out ============================ */

const posts = await loadPosts();

await rm(p("writing"), { recursive: true, force: true });
await mkdir(p("writing"), { recursive: true });
await mkdir(p("about"), { recursive: true });

await writeFile(p("index.html"), homePage(posts));
await writeFile(p("writing", "index.html"), writingIndexPage(posts));
await writeFile(p("about", "index.html"), aboutPage());
await writeFile(p("404.html"), notFoundPage());
await writeFile(p("feed.xml"), feed(posts));

for (const post of posts) {
  await mkdir(p("writing", post.slug), { recursive: true });
  await writeFile(p("writing", post.slug, "index.html"), postPage(post));
}

const sitemap = [
  `${SITE_URL}/`,
  linkOf("writing/"),
  linkOf("about/"),
  ...posts.map((post) => linkOf(post.url)),
]
  .map((url) => `  <url><loc>${url}</loc></url>`)
  .join("\n");
await writeFile(
  p("sitemap.xml"),
  `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${sitemap}\n</urlset>\n`,
);

console.log(
  `Built ${posts.length} post${posts.length === 1 ? "" : "s"}${INCLUDE_DRAFTS ? " (drafts included)" : ""}, ` +
    `${(gh.repos || []).length} repos, ${(gh.pulls || []).filter((x) => x.upstream).length} upstream patches.`,
);

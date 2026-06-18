import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import matter from "gray-matter";

const BLOG_ROOT = process.cwd();
const SOURCE_DIR = path.resolve(process.env.OBSIDIAN_SOURCE_DIR || "..");
const VAULT_SEARCH_DIR = path.resolve(process.env.OBSIDIAN_VAULT_DIR || path.join(SOURCE_DIR, ".."));
const DEFAULT_ATTACHMENT_DIR = path.resolve(SOURCE_DIR, "..", "..", "..", "x", "Attachments");
const POSTS_DIR = path.join(BLOG_ROOT, "src", "content", "posts");
const ASSETS_DIR = path.join(BLOG_ROOT, "public", "notes-assets");
const RECURSIVE = process.argv.includes("--recursive");
const DRY_RUN = process.argv.includes("--dry-run");

const excludedDirs = new Set(["blog", "node_modules", ".git", ".obsidian", ".trash", "dist", ".astro"]);
const excludedSourceNames = new Set(["💻实习具身算法工程师工作日报.md"]);
const excludedAssetNames = new Set(["具身智能仿真平台与算力租赁方案调研报告.pdf"]);
const attachmentExts = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".gif",
  ".svg",
  ".pdf",
  ".mp4",
  ".mov"
]);

const attachmentDirs = uniquePaths(
  (process.env.OBSIDIAN_ATTACHMENT_DIRS || process.env.OBSIDIAN_ATTACHMENT_DIR || "")
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean)
    .concat([DEFAULT_ATTACHMENT_DIR, VAULT_SEARCH_DIR])
    .map((item) => path.resolve(item))
);

async function ensureDir(dir) {
  if (!DRY_RUN) await fs.mkdir(dir, { recursive: true });
}

function uniquePaths(paths) {
  const seen = new Set();
  return paths.filter((item) => {
    const key = item.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function isExcludedSource(filePath) {
  return excludedSourceNames.has(path.basename(filePath));
}

function isExcludedAsset(name) {
  return excludedAssetNames.has(path.basename(name));
}

function slugify(input) {
  const normalized = input
    .normalize("NFKD")
    .replace(/[πΠ]/g, "pi")
    .replace(/[₀⁰]/g, "0")
    .replace(/[₁¹]/g, "1")
    .replace(/[₂²]/g, "2")
    .replace(/[₃³]/g, "3")
    .replace(/[₄⁴]/g, "4")
    .replace(/[₅⁵]/g, "5")
    .replace(/[₆⁶]/g, "6")
    .replace(/[₇⁷]/g, "7")
    .replace(/[₈⁸]/g, "8")
    .replace(/[₉⁹]/g, "9")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();

  return normalized;
}

function hash(value, length = 8) {
  return crypto.createHash("sha1").update(value).digest("hex").slice(0, length);
}

async function listMarkdownFiles(dir, recursive = false) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (recursive && !excludedDirs.has(entry.name)) {
        files.push(...(await listMarkdownFiles(fullPath, recursive)));
      }
      continue;
    }

    if (entry.isFile() && entry.name.toLowerCase().endsWith(".md") && !isExcludedSource(fullPath)) {
      files.push(fullPath);
    }
  }

  return files;
}

async function walkAssets(dir, index = new Map()) {
  let entries = [];
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return index;
  }

  for (const entry of entries) {
    if (entry.isDirectory()) {
      if (!excludedDirs.has(entry.name)) {
        await walkAssets(path.join(dir, entry.name), index);
      }
      continue;
    }

    if (!entry.isFile()) continue;
    if (isExcludedAsset(entry.name)) continue;
    const ext = path.extname(entry.name).toLowerCase();
    if (!attachmentExts.has(ext)) continue;

    const fullPath = path.join(dir, entry.name);
    const key = entry.name.toLowerCase();
    if (!index.has(key)) index.set(key, []);
    index.get(key).push(fullPath);
  }

  return index;
}

function getTitle(content, data, filePath) {
  if (typeof data.title === "string" && data.title.trim()) return data.title.trim();
  const heading = content.match(/^#\s+(.+)$/m);
  if (heading) return heading[1].trim();
  return path.basename(filePath, path.extname(filePath));
}

function normalizeTags(tags) {
  if (!tags) return [];
  if (Array.isArray(tags)) return tags.map(String).map((tag) => tag.trim()).filter(Boolean);
  if (typeof tags === "string") {
    return tags
      .split(/[,\s]+/)
      .map((tag) => tag.trim())
      .filter(Boolean);
  }
  return [];
}

function formatLocalDateKey(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function getDate(data, stat) {
  const value = data.created || data.date || data.pubDate;
  if (value instanceof Date) return formatLocalDateKey(value);
  if (typeof value === "string" && value.trim()) return value.trim().slice(0, 10);
  return formatLocalDateKey(stat.mtime);
}

function stripMarkdown(markdown) {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[\[[^\]]+\]\]/g, " ")
    .replace(/\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]/g, "$2 $1")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[#>*_`~|$\\-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getDescription(content) {
  const withoutImages = content.replace(/^!\[\[[^\]]+\]\]\s*$/gm, "");
  const blocks = withoutImages
    .split(/\n{2,}/)
    .map((block) => stripMarkdown(block))
    .filter((block) => block && !block.startsWith("---"));

  const first = blocks.find((block) => block.length > 30) || blocks[0] || "";
  return first.slice(0, 170);
}

function removeDuplicateTitleHeading(content, title) {
  const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`^\\s*#\\s+${escaped}\\s*\\n+`);
  return content.replace(pattern, "");
}

function normalizeTextInsideMath(content) {
  const map = {
    "π": "pi",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9"
  };

  return content.replace(/\\text\{([^}]*)\}/g, (_, inner) => {
    const normalized = inner.replace(/[π₀₁₂₃₄₅₆₇₈₉]/g, (char) => map[char] ?? char);
    return `\\text{${normalized}}`;
  });
}

function countWords(content) {
  const text = stripMarkdown(content);
  const cjk = text.match(/[\u4e00-\u9fff]/g)?.length ?? 0;
  const latin = text.replace(/[\u4e00-\u9fff]/g, " ").match(/[a-zA-Z0-9_]+/g)?.length ?? 0;
  return cjk + latin;
}

function parseImageSize(parts) {
  const joined = parts.join("|");
  const size = joined.match(/(\d{2,4})x(\d{2,4})/) || joined.match(/(?:^|\|)(\d{2,4})(?:\||$)/);
  if (!size) return "";
  if (size.length === 3) return ` width="${size[1]}" height="${size[2]}"`;
  return ` width="${size[1]}"`;
}

function encodeMarker(payload) {
  return Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
}

function decodeMarker(value) {
  return JSON.parse(Buffer.from(value, "base64url").toString("utf8"));
}

function pickAsset(assetIndex, name) {
  const base = path.basename(name).toLowerCase();
  return assetIndex.get(base)?.[0];
}

async function copyAsset(assetPath, copiedAssets) {
  if (!assetPath) return undefined;
  const ext = path.extname(assetPath);
  const stem = path.basename(assetPath, ext);
  const safeStem = slugify(stem) || `asset-${hash(assetPath, 10)}`;
  let fileName = `${safeStem}${ext.toLowerCase()}`;
  const previous = copiedAssets.get(assetPath);
  if (previous) return previous;

  const existsForOther = Array.from(copiedAssets.values()).includes(fileName);
  if (existsForOther) fileName = `${safeStem}-${hash(assetPath, 6)}${ext.toLowerCase()}`;

  if (!DRY_RUN) await fs.copyFile(assetPath, path.join(ASSETS_DIR, fileName));
  copiedAssets.set(assetPath, fileName);
  return fileName;
}

function normalizeCallouts(content) {
  const labels = {
    summary: "Summary",
    important: "Important",
    note: "Note",
    tip: "Tip",
    warning: "Warning",
    info: "Info"
  };

  return content.replace(/^>\s*\[!(\w+)\]\s*$/gim, (_, type) => {
    const label = labels[type.toLowerCase()] || type;
    return `> **${label}**`;
  });
}

async function transformLinks(content, fileToSlug, assetIndex, copiedAssets) {
  let transformed = normalizeCallouts(content);

  transformed = transformed.replace(/!\[\[([^\]]+)\]\]/g, (_, rawTarget) => {
    const [target, ...parts] = rawTarget.split("|").map((part) => part.trim());
    if (isExcludedAsset(target)) {
      return `<span class="private-asset">未公开附件：${target}</span>`;
    }
    const assetPath = pickAsset(assetIndex, target);
    const alt = path.basename(target);

    if (!assetPath) {
      return `<span class="missing-asset">Missing attachment: ${target}</span>`;
    }

    const size = parseImageSize(parts);
    const marker = `%%ASSET:${encodeMarker({ assetPath, alt, size })}%%`;
    return marker;
  });

  const assetMarkers = [...transformed.matchAll(/%%ASSET:([A-Za-z0-9_-]+)%%/g)];
  for (const marker of assetMarkers) {
    const [full, encoded] = marker;
    const { assetPath, alt, size } = decodeMarker(encoded);
    const fileName = await copyAsset(assetPath, copiedAssets);
    transformed = transformed.replace(
      full,
      `<img src="../../notes-assets/${encodeURIComponent(fileName)}" alt="${alt}" loading="lazy"${size} />`
    );
  }

  transformed = transformed.replace(/\[\[([^\]]+)\]\]/g, (_, rawTarget) => {
    const [targetPart, aliasPart] = rawTarget.split("|").map((part) => part.trim());
    const [targetName, heading] = targetPart.split("#");
    const label = aliasPart || heading || targetName;
    const ext = path.extname(targetName);

    if (ext && attachmentExts.has(ext.toLowerCase())) {
      if (isExcludedAsset(targetName)) {
        return `<span class="private-asset">未公开附件：${targetName}</span>`;
      }
      const assetPath = pickAsset(assetIndex, targetName);
      if (!assetPath) return `<span class="missing-asset">Missing attachment: ${targetName}</span>`;
      const marker = `%%LINKASSET:${encodeMarker({ assetPath, label })}%%`;
      return marker;
    }

    const slug = fileToSlug.get(path.basename(targetName, ".md").toLowerCase());
    if (!slug) return label;

    const anchor = heading ? `#${slugify(heading)}` : "";
    return `[${label}](../${slug}/${anchor})`;
  });

  const linkMarkers = [...transformed.matchAll(/%%LINKASSET:([A-Za-z0-9_-]+)%%/g)];
  for (const marker of linkMarkers) {
    const [full, encoded] = marker;
    const { assetPath, label } = decodeMarker(encoded);
    const fileName = await copyAsset(assetPath, copiedAssets);
    transformed = transformed.replace(full, `[${label}](../../notes-assets/${encodeURIComponent(fileName)})`);
  }

  return transformed;
}

async function cleanGeneratedDirs() {
  if (DRY_RUN) return;
  await fs.rm(POSTS_DIR, { recursive: true, force: true });
  await fs.rm(ASSETS_DIR, { recursive: true, force: true });
  await fs.rm(path.join(BLOG_ROOT, ".astro"), { recursive: true, force: true });
  await ensureDir(POSTS_DIR);
  await ensureDir(ASSETS_DIR);
}

async function main() {
  await ensureDir(POSTS_DIR);
  await ensureDir(ASSETS_DIR);

  const markdownFiles = await listMarkdownFiles(SOURCE_DIR, RECURSIVE);
  const assetIndex = new Map();
  for (const dir of attachmentDirs) {
    await walkAssets(dir, assetIndex);
  }
  const fileToSlug = new Map();
  const slugs = new Set();

  for (const file of markdownFiles) {
    const raw = await fs.readFile(file, "utf8");
    const parsed = matter(raw);
    const title = getTitle(parsed.content, parsed.data, file);
    const date = getDate(parsed.data, await fs.stat(file));
    let slug = slugify(path.basename(file, ".md")) || slugify(title) || `note-${date}`;
    if (slug.endsWith("-obsidian")) slug = slug.slice(0, -9);
    if (!slug) slug = `note-${hash(file, 6)}`;
    const baseSlug = slug;
    let counter = 2;
    while (slugs.has(slug)) slug = `${baseSlug}-${counter++}`;
    slugs.add(slug);
    fileToSlug.set(path.basename(file, ".md").toLowerCase(), slug);
  }

  await cleanGeneratedDirs();

  const copiedAssets = new Map();
  const written = [];

  for (const file of markdownFiles) {
    const raw = await fs.readFile(file, "utf8");
    const stat = await fs.stat(file);
    const parsed = matter(raw);
    const title = getTitle(parsed.content, parsed.data, file);
    const sourceBase = path.basename(file, ".md").toLowerCase();
    const slug = fileToSlug.get(sourceBase);
    const date = getDate(parsed.data, stat);
    const tags = normalizeTags(parsed.data.tags);
    const publishContent = normalizeTextInsideMath(
      removeDuplicateTitleHeading(parsed.content.trim(), title)
    );
    const transformed = await transformLinks(publishContent, fileToSlug, assetIndex, copiedAssets);
    const words = countWords(transformed);
    const output = matter.stringify(`${transformed}\n`, {
      title,
      description: parsed.data.description || getDescription(parsed.content),
      pubDate: date,
      updatedDate: formatLocalDateKey(stat.mtime),
      tags,
      draft: Boolean(parsed.data.draft),
      source: path.relative(SOURCE_DIR, file).replaceAll("\\", "/"),
      wordCount: words,
      readingTime: Math.max(1, Math.ceil(words / 500))
    });

    const target = path.join(POSTS_DIR, `${slug}.md`);
    if (!DRY_RUN) await fs.writeFile(target, output, "utf8");
    written.push({ slug, title });
  }

  console.log(`Synced ${written.length} notes from ${SOURCE_DIR}`);
  if (copiedAssets.size) console.log(`Copied ${copiedAssets.size} attachments to ${ASSETS_DIR}`);
  const missing = written.length === 0 ? "No markdown files found." : "";
  if (missing) console.warn(missing);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

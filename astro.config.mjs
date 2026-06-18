import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import { rehypeMermaidCode } from "./src/lib/rehype-mermaid-code.mjs";

const site = process.env.SITE_URL || "https://lilyuan258.github.io";
const base = process.env.SITE_BASE || "/";

export default defineConfig({
  site,
  base,
  integrations: [mdx(), sitemap()],
  markdown: {
    syntaxHighlight: "shiki",
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [rehypeKatex, { strict: false }],
      rehypeMermaidCode,
      rehypeSlug,
      [
        rehypeAutolinkHeadings,
        {
          behavior: "wrap",
          properties: {
            className: ["heading-anchor"],
            ariaHidden: false
          }
        }
      ]
    ]
  },
  vite: {
    build: {
      chunkSizeWarningLimit: 900
    }
  }
});

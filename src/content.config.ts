import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const posts = defineCollection({
  loader: glob({
    pattern: "*.md",
    base: "./src/content/posts"
  }),
  schema: z
    .object({
      title: z.string(),
      description: z.string().default(""),
      pubDate: z.coerce.date(),
      updatedDate: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]),
      draft: z.boolean().default(false),
      source: z.string().optional(),
      wordCount: z.number().optional(),
      readingTime: z.number().optional()
    })
    .passthrough()
});

export const collections = { posts };

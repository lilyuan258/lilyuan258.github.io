import type { CollectionEntry } from "astro:content";

export type Post = CollectionEntry<"posts">;

export function sortPosts(posts: Post[]) {
  return [...posts].sort(
    (a, b) => b.data.pubDate.getTime() - a.data.pubDate.getTime()
  );
}

export function formatDate(date: Date) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "short",
    day: "numeric"
  }).format(date);
}

export function getAllTags(posts: Post[]) {
  return Array.from(new Set(posts.flatMap((post) => post.data.tags ?? []))).sort(
    (a, b) => a.localeCompare(b)
  );
}

export function getTotalWords(posts: Post[]) {
  return posts.reduce((sum, post) => sum + (post.data.wordCount ?? 0), 0);
}

export function getLatestDate(posts: Post[]) {
  if (!posts.length) return undefined;
  return sortPosts(posts)[0].data.updatedDate ?? sortPosts(posts)[0].data.pubDate;
}

export function compactNumber(value: number) {
  if (value >= 10000) {
    return `${(value / 10000).toFixed(value >= 100000 ? 0 : 1)}w`;
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k`;
  }
  return String(value);
}

export function daysBetween(startDate: string, end = new Date()) {
  const start = new Date(`${startDate}T00:00:00`);
  const diff = end.getTime() - start.getTime();
  return Math.max(1, Math.ceil(diff / 86_400_000));
}

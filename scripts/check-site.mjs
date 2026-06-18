import { chromium, expect } from "@playwright/test";

const BASE_URL = process.env.SITE_CHECK_URL || "http://127.0.0.1:4321";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });

try {
  await page.goto(BASE_URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await expect(page.getByRole("heading", { name: "Li Yuan" })).toBeVisible();
  await expect(page.getByRole("link", { name: "GitHub" }).first()).toBeVisible();
  await expect(page.getByText("最新笔记")).toBeVisible();

  await page.goto(`${BASE_URL}/blog/`, { waitUntil: "domcontentloaded", timeout: 60_000 });
  await expect(page.getByRole("heading", { name: "Notes" })).toBeVisible();
  await expect(page.locator(".post-card")).toHaveCount(5);
  await page.locator("[data-search]").fill("PaliGemma");
  await expect(page.locator(".post-card:not([hidden])")).toHaveCount(1);

  await page.goto(`${BASE_URL}/blog/paligemma-vlm-obsidian-notes/`, {
    waitUntil: "domcontentloaded",
    timeout: 60_000
  });
  await expect(page.getByRole("heading", { name: "PaliGemma VLM 学习笔记" })).toBeVisible();
  await expect(page.locator(".markdown-body")).toBeVisible();

  await page.getByLabel("切换明暗主题").click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");

  await page.screenshot({ path: "dist/site-check-home.png", fullPage: true });
} finally {
  await browser.close();
}

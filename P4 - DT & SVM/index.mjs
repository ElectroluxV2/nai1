import puppeteer from 'puppeteer';
import { EOL } from 'os';
import { promises as fs } from 'fs';

// Launch the browser and open a new blank page
const browser = await puppeteer.launch({
  headless: false,
});

const file = await fs.open('top-50-websites-dark-mode.csv', 'w');

try {
  const [ page ] = await browser.pages();

  // Set screen size
  await page.setViewport({width: 1080, height: 1024});

  console.log('Loading top 50 websites');

  // Navigate the page to a URL
  await page.goto('https://www.similarweb.com/top-websites/');
  // Wait for a page load
  await page.waitForSelector('.top-table__body');

  const table = await page.evaluate(() => Array.from(document.querySelectorAll('tbody > tr')).map(tr => Array.from(tr.getElementsByTagName('td')).map(td => td.textContent)));

  console.log(`Loaded ${table.length} websites`);

  await file.truncate();

  // For each page, check if it has dark mode support
  for (const [rank, name, _category, _change, avgVisitDuration] of table) {
    console.log(`Evaluating ${name}`);

    const validUrl = name.startsWith('http') ? name : 'https://' + name;
    try {
      await page.goto(validUrl);
    } catch {
      console.warn('Failed to load, skipping');
      continue;
    }

    // Emulate dark mode preference
    await page.emulateMediaFeatures([{
      name: 'prefers-color-scheme',
      value: 'dark',
    }]);

    // Give 1s, so page may load some JS magic
    await new Promise(r => setTimeout(r, 1_000));

    // Proper dark mode support must include `color-scheme=*dark` CSS property on HTML tag
    // const hasDarkColorScheme = await page.evaluate(() => document.getElementsByTagName('html')[0].style.colorScheme?.includes('dark'));
    // However only twitter.com is doing it the right way, so lets assume that mention of dark anywhere is sign of supported dark mode
    const hasDarkMode = await page.evaluate(() => document.documentElement.outerHTML.includes('dark'))
    console.log('Has dark mode support', hasDarkMode);

    const row = [
      name,
      rank,
      new Date(`Jan 01 1970 ${avgVisitDuration} GMT`).getTime() / 1000, // Convert to seconds
      Number(hasDarkMode),
    ];

    await file.appendFile(row.join(',') + EOL);
  }

  await new Promise(r => setTimeout(r, 15_000));
} finally {
  await file.close();
  await browser.close();
}

/**
 * Tools collection structured as entries
 * 
 * Each entry has the following structure:
 * {
 *     title: string // display name
 *     link: string // the link for the website which contain learning resources
 *     description?: string // description that will be listed with your entry
 *     github?: string // username on github that will display a link to your repo
 *     tag: string // category/tag of the tool
 * }
 */

import { llmEntries } from './llm-entries.js';
import { aiEntries } from './ai-entries.js';
import { mcpEntries } from './mcp-entries.js';
import { lfaiEntries } from './lfai-entries.js';

// Combine all entries with hosted LF AI & Data projects prioritized
const seen = new Set();
const all = [...lfaiEntries, ...mcpEntries, ...llmEntries, ...aiEntries];

export const entries = all.filter(entry => {
  const key = (entry.title || "").toLowerCase().trim();
  if (!key || seen.has(key)) return false;
  seen.add(key);
  return true;
});


---
name: writer
model:
  primary: gpt-4o
  fallbacks: [gpt-4o-mini]
tools: [file, terminal]
middleware:
  - budget
---
You are a careful writer. Be concise.

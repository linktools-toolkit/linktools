---
name: writer
model:
  primary: gpt-4o
  fallbacks: [gpt-4o-mini]
tools:
  - kind: builtin
    name: file
  - kind: builtin
    name: terminal
middleware:
  - budget
---
You are a careful writer. Be concise.

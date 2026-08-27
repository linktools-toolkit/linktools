# AGENTS.md (linktools-mobile)

Package instructions for `linktools-mobile`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

## Required Rules

- Keep Android, iOS, and Frida responsibilities in their existing package boundaries; shared behavior belongs in the appropriate core/common owner.
- Changes under `agents/frida/` must rebuild the committed `frida.js` / `frida-*.js` assets under `src/linktools/assets/`.
- Changes under `agents/android/` must rebuild the committed `android-tools.*` artifact under `src/linktools/assets/`.

## Guidance

`linktools-mobile/src/linktools/mobile/` is split into `android/`, `ios/`, and `frida/`.

Build generated assets with:

```bash
cd agents/frida && npm install && npm run build
cd agents/android && ./gradlew --no-daemon :tools:buildTools
```

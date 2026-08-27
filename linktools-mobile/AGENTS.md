# AGENTS.md (linktools-mobile)

Package instructions for `linktools-mobile`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

## Required Rules

- Committed generated mobile assets must stay in sync with their source projects.

## Guidance

`linktools-mobile/src/linktools/mobile/` is currently split into `android/`, `ios/`, and `frida/`.

When the corresponding sources change, rebuild the committed assets with:

```bash
cd agents/frida && npm install && npm run build
cd agents/android && ./gradlew --no-daemon :tools:buildTools
```

The Frida build updates `frida.js` / `frida-*.js`; the Android build updates `android-tools.*` under `src/linktools/assets/`.

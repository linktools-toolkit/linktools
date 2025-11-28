const { join, resolve } = require("path")
const { mkdir } = require("fs");
const { execSync } = require("node:child_process");

const sourcePath = resolve("./");
const targetPath = join(sourcePath, "../../src/linktools/assets");

function exec(cmd) {
    console.log("exec cmd: " + cmd);
    execSync(cmd, { stdio: "inherit" });
}

function compile(input, output, options = void 0) {
    const getOption = function (key, defaultValue = void 0) {
        if (options !== void 0) {
            const value = options[key];
            if (value !== void 0) {
                return value;
            }
        }
        return defaultValue;
    }
    let cmd = `frida-compile ${input} -o ${output}`;
    if (getOption('watch', false)) {
        cmd += " -w";
    }
    if (getOption("compress", false)) {
        cmd += " -c";
    }
    const bundleFormat = getOption('bundleFormat');
    if (bundleFormat && bundleFormat.trim() !== "") {
        cmd += ` -B ${bundleFormat}`;
    }
    exec(cmd);
}

function uglifyjs(input, output) {
    exec(`uglifyjs ${input} --mangle --output ${output}`);
}

function compileBridge() {
    const tempPath = join(sourcePath, "build");
    mkdir(tempPath, { recursive: true }, (err) => { });
    const compileOne = function (name) {
        compile(`${sourcePath}/${name}.js`, `${tempPath}/${name}.js`, { compress: true, bundleFormat: 'iife' });
        uglifyjs(`${tempPath}/${name}.js`, `${targetPath}/${name}.js`);
    }
    compileOne('frida-java-bridge');
    compileOne('frida-objc-bridge');
}

function compileScript(debug) {
    compile(`${sourcePath}/index.ts`, `${targetPath}/frida.js`, { watch: debug, compress: false, bundleFormat: 'iife' });
    if (!debug) {
        uglifyjs(`${targetPath}/frida.js`, `${targetPath}/frida.min.js`);
    }
}

if (process.env.FRIDA_BUILD_DEBUG === "1") {
    compileBridge();
    compileScript(true);
} else {
    compileBridge();
    compileScript(false);
}

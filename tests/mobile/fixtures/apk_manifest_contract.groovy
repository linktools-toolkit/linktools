import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

class CopySpec {
    String source
    String destination
    String targetName
    void from(Object value) { source = value.toString() }
    void into(Object value) { destination = value.toString() }
    void rename(String original, String target) { targetName = target }
}

class BuildToolsTask {
    Object dependsOn
    Closure action
    void dependsOn(Object value) { dependsOn = value }
    void doLast(Closure value) { action = value }
}

File source = new File(args[0])
File root = new File(args[1], 'mobile/agents/android')
File build = new File(root, 'tools/build')
File apk = new File(build, 'outputs/apk/release/tools-release-unsigned.apk')
File assets = new File(root, '../../src/linktools/assets').canonicalFile
File target = new File(assets, 'android-tools.apk')
File manifest = new File(assets, 'android-tools.json')
apk.parentFile.mkdirs()
assets.mkdirs()
BuildToolsTask task = new BuildToolsTask()
int copies = 0
Binding binding = new Binding([
    rootDir: root,
    project: [buildDir: build],
    buildTools: task,
    plugins: { Closure ignored -> },
    apply: { Map ignored -> },
    android: { Closure ignored -> },
    dependencies: { Closure ignored -> },
    file: { Object path -> new File(path.toString()) },
    copy: { Closure body ->
        CopySpec spec = new CopySpec()
        body.delegate = spec
        body.resolveStrategy = Closure.DELEGATE_FIRST
        body.call()
        Files.copy(new File(spec.source).toPath(), new File(spec.destination, spec.targetName).toPath(), StandardCopyOption.REPLACE_EXISTING)
        copies++
    },
    tasks: [register: { Object... values ->
        if (values[0] == 'buildTools') {
            Closure body = values[-1]
            body.delegate = task
            body.resolveStrategy = Closure.DELEGATE_FIRST
            body.call()
        }
    }],
])
// Only Gradle's DSL and copy service are replaced; the complete build script is evaluated unchanged.
new GroovyShell(binding).evaluate('class Jar {}\n' + source.text, 'build.gradle')
assert task.dependsOn == ':tools:assembleRelease'
assert task.action != null

def md5 = { byte[] content -> MessageDigest.getInstance('MD5').digest(content).encodeHex().toString() }
def run = { task.action.call() }
def read = { new JsonSlurper().parse(manifest) }
def initial = [name: 'android-tools.apk', md5: md5('first'.bytes), main: 'android.tools.Main', size: 5, time: 'unchanged timestamp']
apk.bytes = 'first'.bytes
target.bytes = apk.bytes
manifest.text = JsonOutput.toJson([AGENT_APK: initial, FRIDA_SERVER: [keep: true]]) + '\n'
byte[] previous = manifest.bytes
long modified = manifest.lastModified()
run()
assert copies == 0
assert manifest.bytes == previous
assert manifest.lastModified() == modified
assert target.bytes == apk.bytes

apk.bytes = 'second apk version'.bytes
run()
assert copies == 1
assert target.bytes == apk.bytes
assert read().AGENT_APK.name == 'android-tools.apk'
assert read().AGENT_APK.main == 'android.tools.Main'
assert read().AGENT_APK.md5 == md5(apk.bytes)
assert read().AGENT_APK.size == apk.length()
assert read().AGENT_APK.time != initial.time
assert read().FRIDA_SERVER == [keep: true]
previous = manifest.bytes
modified = manifest.lastModified()
run()
assert copies == 1
assert manifest.bytes == previous
assert manifest.lastModified() == modified

// A matching manifest cannot make a missing or damaged copied APK up-to-date.
for (boolean missing : [true, false]) {
    assert read().AGENT_APK.md5 == md5(apk.bytes)
    if (missing) {
        assert target.delete()
    } else {
        target.bytes = new byte[(int) apk.length()]
        assert target.length() == apk.length()
        assert md5(target.bytes) != md5(apk.bytes)
    }
    int before = copies
    run()
    assert copies == before + 1
    assert target.bytes == apk.bytes
    assert read().AGENT_APK.md5 == md5(apk.bytes)
    assert read().AGENT_APK.size == target.length()
    previous = manifest.bytes
    modified = manifest.lastModified()
    long targetModified = target.lastModified()
    run()
    assert copies == before + 1
    assert manifest.bytes == previous
    assert manifest.lastModified() == modified
    assert target.lastModified() == targetModified
}

// Missing canonical metadata must rebuild even if the obsolete key has a matching checksum.
for (Map config : [[:], [tools_apk: [md5: md5(apk.bytes)]], [AGENT_APK: [name: 'incomplete']]]) {
    manifest.text = JsonOutput.toJson(config)
    int before = copies
    run()
    assert copies == before + 1
    assert read().AGENT_APK.md5 == md5(apk.bytes)
    assert read().AGENT_APK.size == apk.length()
    previous = manifest.bytes
    run()
    assert copies == before + 1
    assert manifest.bytes == previous
}
manifest.delete()
int before = copies
run()
assert copies == before + 1
assert read().AGENT_APK.md5 == md5(apk.bytes)
println('APK manifest contract passed')

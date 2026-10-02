const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const emitted = [];
const timers = new Map();
let timerId = 0;
const context = {
    rpc: {}, console: {}, ModuleMap: class {}, NativePointer: class {},
    setTimeout: fn => { timers.set(++timerId, fn); return timerId; },
    clearTimeout: id => timers.delete(id),
    send: payload => emitted.push(...payload.$events.map(event => event.msg)),
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);

function flush() {
    for (const fn of Array.from(timers.values())) fn();
    assert.equal(timers.size, 0);
    assert.equal(emitted.length, 1);
    return JSON.parse(JSON.stringify(emitted.pop()));
}

let cases = 0;
for (const name of ['JavaHelper', 'ObjCHelper', 'CHelper']) {
    for (const args of [undefined, false, true]) {
        for (const result of [undefined, false, true]) {
            for (const error of [undefined, false, true]) {
                for (const throws of [false, true]) {
                    for (const extras of [undefined, { tag: 'extra', count: 7 }]) {
                        const options = { method: false, thread: false, stack: false, page: false };
                        for (const [key, value] of Object.entries({ args, result, error, extras })) {
                            if (value !== undefined) options[key] = value;
                        }
                        const callback = context[name].getEventImpl(options);
                        const object = { $className: 'Example' };
                        const arguments_ = [42, 'argument'];
                        const returned = vm.runInContext('({ sentinel: "return" })', context);
                        const failure = vm.runInContext('new Error("original failure")', context);
                        let calls = 0;
                        const original = function (...received) {
                            calls++;
                            assert.deepEqual(received, name === 'CHelper' ? [arguments_] : [object, arguments_]);
                            if (throws) throw failure;
                            return returned;
                        };
                        const invoke = () => name === 'CHelper'
                            ? callback.call(original, arguments_)
                            : callback.call(original, object, arguments_);
                        if (throws) assert.throws(invoke, value => value === failure);
                        else assert.equal(invoke(), returned);
                        assert.equal(calls, 1);
                        const expected = { ...extras };
                        if (args === true) expected.args = arguments_;
                        if (result === true || (result === undefined && args === true)) {
                            expected.result = throws ? null : '[object Object]';
                        }
                        if (error === true || (error === undefined && args === true)) {
                            expected.error = throws ? 'Error: original failure' : null;
                        }
                        assert.deepEqual(flush(), expected, `${name} ${JSON.stringify(options)} throws=${throws}`);
                        cases++;
                    }
                }
            }
        }
    }
}
// Extras are copied when the hook is created; explicit data fields overwrite matching extras.
for (const name of ['JavaHelper', 'ObjCHelper', 'CHelper']) {
    const extras = { tag: 'before', args: 'extra-args', result: 'extra-result', error: 'extra-error' };
    const callback = context[name].getEventImpl({
        method: false, args: true, result: true, error: true, extras,
    });
    extras.tag = 'after';
    const original = () => 'original-result';
    if (name === 'CHelper') callback.call(original, []);
    else callback.call(original, {}, []);
    assert.deepEqual(flush(), { tag: 'before', args: [], result: 'original-result', error: null });
    cases++;
}
// Native onLeave does not invoke the original; its result flag still follows args by default.
for (const args of [undefined, false, true]) {
    for (const result of [undefined, false, true]) {
        const callback = context.CHelper.getEventImpl({ method: false, args, result, extras: { tag: 'leave' } });
        callback.onLeave.call({}, 23);
        const expected = { tag: 'leave' };
        if (result === true || (result === undefined && args === true)) expected.result = 23;
        assert.deepEqual(flush(), expected);
        cases++;
    }
}
process.stdout.write(JSON.stringify({ cases }));

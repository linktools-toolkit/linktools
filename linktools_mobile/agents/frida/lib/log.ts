export const DEBUG = 1;
export const INFO = 2;
export const WARNING = 3;
export const ERROR = 4;
let $level = INFO;

let $pendingEvents: any[] = [];
let $flushTimer: any = null;

export function getLevel(): number {
    return $level;
}

export function setLevel(level: number) {
    $level = level;
    d("Set log level: " + level);
}

export function d(message: any, data?: ArrayBuffer | number[] | null) {
    if ($level <= DEBUG) {
        $send("log", { level: "debug", message: message }, data);
    }
}

export function i(message: any, data?: ArrayBuffer | number[] | null) {
    if ($level <= INFO) {
        $send("log", { level: "info", message: message }, data);
    }
}

export function w(message: any, data?: ArrayBuffer | number[] | null) {
    if ($level <= WARNING) {
        $send("log", { level: "warning", message: message }, data);
    }
}

export function e(message: any, data?: ArrayBuffer | number[] | null) {
    if ($level <= ERROR) {
        $send("log", { level: "error", message: message }, data);
    }
}

export function event(message: { [name: string]: any; }, data?: ArrayBuffer | number[] | null) {
    $send("msg", message, data);
}

export function exception(description: string, stack: string) {
    $send("error", {description: description, stack: stack});
}

function $send(type: string, message: any, data?: ArrayBuffer | number[] | null) {
    const event: { [name: string]: any; } = {};
    event[type] = message;

    if (data == null) {
        // 如果data为空，则加到pending队列，打包一起发送
        $pendingEvents.push(event);
        if ($pendingEvents.length >= 50) {
            // 当短时间积累的事件太多，可能会出现卡死的情况
            // 所以设置一个pending队列的阈值
            $flush();
        } else if ($flushTimer === null) {
            $flushTimer = setTimeout($flush, 50);
        }
    } else {
        // data不为空，就不能一次性发送多个event
        // 立即把pending队列的发过去，然后发送带data的message
        $flush();
        send({ $events: [event] }, data);
    }
}

function $flush() {
    if ($flushTimer !== null) {
        clearTimeout($flushTimer);
        $flushTimer = null;
    }

    if ($pendingEvents.length === 0) {
        return;
    }

    const events = $pendingEvents;
    $pendingEvents = [];

    send({ $events: events });
}

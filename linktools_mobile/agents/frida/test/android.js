Java.perform(function () {

    AndroidHelper.bypassSslPinning();
    AndroidHelper.setWebviewDebuggingEnabled();

    // spdy test
    ignoreError(() => JavaHelper.hookMethods(
        "anet.channel.entity.ConnType",
        "isHttpType",
        () => true
    ));

    // rpc test
    ignoreError(() => JavaHelper.hookMethods(
        "com.alipay.mobile.common.transport.http.HttpUrlRequest",
        "isRpcHttp2",
        () => false
    ));

});

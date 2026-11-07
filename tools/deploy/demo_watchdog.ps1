# 小蟹演示站看门狗：每 5 分钟确认一次站点真的能回 200，不行就重启服务。
#
# 为什么要它：XiaoxieDemo 任务只有"开机时启动"这一个触发器，python 进程一旦
# 退出（被系统终止、被安全软件结束、自己崩了），任务就一直停在 Ready，页面
# 从此 404 到下一次有人发现。2026-10-29 那次就是这么停了六天。
#
# 只检查 TCP 监听是不够的：端口还占着但进程已经挂死（不响应请求）时，
# 监听状态是正常的，页面却打不开。所以这里用本机 HTTP 请求判断。

$ErrorActionPreference = "SilentlyContinue"
$log = "C:\demo\watchdog.log"

function Write-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File -FilePath $log -Append -Encoding utf8
}

$ok = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8443/" -UseBasicParsing -TimeoutSec 8
    $ok = ($r.StatusCode -eq 200)
} catch {
    $ok = $false
}

if ($ok) { exit 0 }

Write-Log "站点无响应，开始重启"

# 残留进程会占着 8443：先清掉，否则新实例起不来
$conns = Get-NetTCPConnection -LocalPort 8443 -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    if ($c.OwningProcess -and $c.OwningProcess -ne 0) {
        Write-Log "结束占用 8443 的进程 PID=$($c.OwningProcess)"
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 2

Start-ScheduledTask -TaskName "XiaoxieDemo"
Start-Sleep -Seconds 6

$ok2 = $false
try {
    $r2 = Invoke-WebRequest -Uri "http://127.0.0.1:8443/" -UseBasicParsing -TimeoutSec 10
    $ok2 = ($r2.StatusCode -eq 200)
} catch {
    $ok2 = $false
}

if ($ok2) {
    Write-Log "已恢复（HTTP 200）"
} else {
    Write-Log "重启后仍然不通，需要人工看：Get-ScheduledTaskInfo -TaskName XiaoxieDemo"
}
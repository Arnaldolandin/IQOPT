' Arranca el watchdog del bot IQOPT en segundo plano, sin ventana de consola.
' Para auto-arranque: crear un acceso directo a este .vbs en
'   shell:startup  (Win+R -> shell:startup)
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "D:\Proyects\IQOPT"
WshShell.Run "D:\Proyects\IQOPT\run_watchdog.bat", 0, False

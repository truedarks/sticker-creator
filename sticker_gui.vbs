Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

REM Set current directory to script location
WshShell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)

REM Check if Python is available
On Error Resume Next
Set pythonCheck = WshShell.Exec("python --version")
pythonCheck.StdOut.ReadAll
pythonCheck.StdErr.ReadAll
If Err.Number <> 0 Or pythonCheck.ExitCode <> 0 Then
    MsgBox "Python is not found in PATH." & vbCrLf & "Please install Python or add it to your PATH.", 48, "Error"
    WScript.Quit
End If
On Error Goto 0

REM Launch GUI without showing console window (0 = hidden)
WshShell.Run "python sticker_gui.py", 0, False

Set WshShell = Nothing
Set fso = Nothing

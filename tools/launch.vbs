Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
toolsDir = fso.GetParentFolderName(WScript.ScriptFullName)
appScript = fso.BuildPath(toolsDir, "publisher-app.cjs")
nodeExe = "C:\Program Files\nodejs\node.exe"
If Not fso.FileExists(nodeExe) Then
  nodeExe = "node"
End If
cmd = """" & nodeExe & """ """ & appScript & """"
If WScript.Arguments.Count > 0 Then
  If Len(Trim(WScript.Arguments(0))) > 0 Then
    cmd = cmd & " """ & WScript.Arguments(0) & """"
  End If
End If
WshShell.Run cmd, 0, False

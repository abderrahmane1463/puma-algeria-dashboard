$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open("c:\Users\Education Bouira\Documents\stnns\RPPRT-STG.doc")
    $text = $doc.Content.Text
    $text | Out-File -FilePath "c:\Users\Education Bouira\Documents\stnns\RPPRT-STG.txt" -Encoding utf8
    $doc.Close()
} finally {
    $word.Quit()
}

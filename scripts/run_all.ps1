# Train, evaluate and render figures end to end on Windows (run with the project venv active).
# Output is mirrored to train.log.
$env:PYTHONIOENCODING = 'utf-8'

oncoscope train *>&1 | Tee-Object -FilePath train.log
oncoscope evaluate *>&1 | Tee-Object -FilePath train.log -Append
python scripts\gallery.py *>&1 | Tee-Object -FilePath train.log -Append
Rscript r\figures.R *>&1 | Tee-Object -FilePath train.log -Append
Write-Output 'RUN_ALL_FINISHED' | Tee-Object -FilePath train.log -Append

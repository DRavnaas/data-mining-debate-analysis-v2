@if .%ZKCLI_HOME%. == .. set ZKCLI_HOME=%solr_home%\server\scripts\cloud-scripts
@set collection=%1
@if .%1. == .. set collection=gettingstarted
@set cfgdir=%2
@if .%2. == .. set cfgdir=curcfg_%collection%
@set zkhost=%3
@if .%3. == .. set zkhost=localhost:9983

@set fileerr=1
@if exist %cfgdir%\schema.xml if exist %cfgdir%\solrconfig.xml set fileerr=
@if .%fileerr%. == .1. @echo Could not find %cfgdir%\schema.xml or %cfgdir%\solrconfig.xml
@if .%fileerr%. == .1. goto :end
@echo.

@echo.
@echo Uploading config for collection %collection% from directory %cfgdir%
@echo.

call %solr_home%\server\scripts\cloud-scripts\zkcli -zkhost %zkhost% -cmd upconfig -confdir %cfgdir% -confname %collection%

@Echo Do you want to clean the collection of current documents from %collection%?
@CHOICE /C YN /D N /T 5
@if errorlevel 2 goto end

curl "http://localhost:8983/solr/%collection%/update?stream.body=%3Cdelete%3E%3Cquery%3E*:*%3C/query%3E%3C/delete%3E&commit=true"

@rem to do - add a small wait and then delete all (first url) then reload (second url - maybe not necessary after a delete?)
@rem http://localhost:8983/solr/csvtest/update?stream.body=%3Cdelete%3E%3Cquery%3E*:*%3C/query%3E%3C/delete%3E&commit=true
@rem http://localhost:8983/solr/admin/collections?action=RELOAD&name=csvtest&reindex=true&deleteAll=true

:end
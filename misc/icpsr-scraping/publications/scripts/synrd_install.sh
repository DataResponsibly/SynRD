#!/bin/bash
pip3 install selenium
apt-get update # to update ubuntu to correctly run apt install
apt install chromium-chromedriver
cp /usr/lib/chromium-browser/chromedriver /usr/bin

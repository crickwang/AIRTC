#!/bin/bash
cd /home/ubuntu/AIRTC
git pull origin main
pip install -r requirements.txt --quiet
pm2 restart all
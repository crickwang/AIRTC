#!/bin/bash
cd /home/ubuntu/AIRTC
git pull origin main
pm2 restart all
#! /usr/bin/sh
python pipeline.py | tee logs/log-$(date '+%Y-%m-%d-%H-%M').txt
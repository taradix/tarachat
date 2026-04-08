#!/bin/sh

# Configuration
LE_DIR=/etc/letsencrypt

# Copy primary cert to canonical paths used by nginx
cp ${LE_DIR}/live/${HOSTNAME}/privkey.pem ${LE_DIR}/live/privkey.pem
cp ${LE_DIR}/live/${HOSTNAME}/fullchain.pem ${LE_DIR}/live/fullchain.pem

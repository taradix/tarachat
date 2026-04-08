#!/bin/sh

# Configuration
LE_DIR=/etc/letsencrypt
DHPARAMS_PEM=${LE_DIR}/dhparams.pem

# Create DH parameters if it doesn't exist already - needed by Dovecot.
if [ ! -e "${DHPARAMS_PEM}" ]; then
  openssl dhparam -out ${DHPARAMS_PEM} 2048
fi

# Copy primary private key and full chain to live directory
cp ${LE_DIR}/live/${SERVER_HOSTNAME}/privkey.pem ${LE_DIR}/live/privkey.pem
cp ${LE_DIR}/live/${SERVER_HOSTNAME}/fullchain.pem ${LE_DIR}/live/fullchain.pem

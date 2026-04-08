#!/bin/sh

# Exit on any error
set -eu

# Configuration
LE_DIR="/etc/letsencrypt"
LIVE_DIR="${LE_DIR}/live/${HOSTNAME}"
ARCHIVE_DIR="${LE_DIR}/archive/${HOSTNAME}"

if [ -e "${LIVE_DIR}/fullchain.pem" ] && [ -e "${LIVE_DIR}/privkey.pem" ]; then
  echo "Reusing existing certificate for ${HOSTNAME}!"
  exit 0
fi

mkdir -p "${LIVE_DIR}" "${ARCHIVE_DIR}"

# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "${ARCHIVE_DIR}/privkey1.pem" \
  -out "${ARCHIVE_DIR}/cert1.pem" \
  -subj "/CN=${HOSTNAME}"

# Create chain and fullchain
cp "${ARCHIVE_DIR}/cert1.pem" "${ARCHIVE_DIR}/chain1.pem"
cat "${ARCHIVE_DIR}/cert1.pem" "${ARCHIVE_DIR}/chain1.pem" > "${ARCHIVE_DIR}/fullchain1.pem"

# Create symlinks in live directory
ln -sf "../../archive/${HOSTNAME}/cert1.pem" "${LIVE_DIR}/cert.pem"
ln -sf "../../archive/${HOSTNAME}/chain1.pem" "${LIVE_DIR}/chain.pem"
ln -sf "../../archive/${HOSTNAME}/fullchain1.pem" "${LIVE_DIR}/fullchain.pem"
ln -sf "../../archive/${HOSTNAME}/privkey1.pem" "${LIVE_DIR}/privkey.pem"

echo "Generated self-signed certificate for ${HOSTNAME}!"

#!/bin/sh

# Exit on any error
set -eu

# Configuration
LE_DIR="/etc/letsencrypt"
LIVE_DIR="${LE_DIR}/live/${SERVER_HOSTNAME}"
ARCHIVE_DIR="${LE_DIR}/archive/${SERVER_HOSTNAME}"
RENEWAL_DIR="${LE_DIR}/renewal"

if [ -e "${LIVE_DIR}/fullchain.pem" ] && [ -e "${LIVE_DIR}/privkey.pem" ]; then
  echo "Reusing existing certificate for ${SERVER_HOSTNAME}!"
  exit 0
fi

mkdir -p "${LIVE_DIR}" "${ARCHIVE_DIR}" "${RENEWAL_DIR}"

# Generate private key and self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout "${ARCHIVE_DIR}/privkey1.pem" -out "${ARCHIVE_DIR}/cert1.pem" -subj "/CN=${SERVER_HOSTNAME}" -addext "subjectAltName=IP:${IPV4_NETWORK}.240"

# Create chain.pem (self-signed so chain = cert) and fullchain.pem (cert + chain)
cp "${ARCHIVE_DIR}/cert1.pem" "${ARCHIVE_DIR}/chain1.pem"
cat "${ARCHIVE_DIR}/cert1.pem" "${ARCHIVE_DIR}/chain1.pem" > "${ARCHIVE_DIR}/fullchain1.pem"

# Create symlinks in live directory
ln -sf "../../archive/${SERVER_HOSTNAME}/cert1.pem" "${LIVE_DIR}/cert.pem"
ln -sf "../../archive/${SERVER_HOSTNAME}/chain1.pem" "${LIVE_DIR}/chain.pem"
ln -sf "../../archive/${SERVER_HOSTNAME}/fullchain1.pem" "${LIVE_DIR}/fullchain.pem"
ln -sf "../../archive/${SERVER_HOSTNAME}/privkey1.pem" "${LIVE_DIR}/privkey.pem"

# Create a mock renewal config
cat <<EOF > "${RENEWAL_DIR}/${SERVER_HOSTNAME}.conf"
# Mock renewal configuration
version = 1.0.0
archive_dir = ${ARCHIVE_DIR}
cert = ${LIVE_DIR}/cert.pem
privkey = ${LIVE_DIR}/privkey.pem
chain = ${LIVE_DIR}/chain.pem
fullchain = ${LIVE_DIR}/fullchain.pem
EOF

echo "Generated self-signed certificate for ${SERVER_HOSTNAME}!"

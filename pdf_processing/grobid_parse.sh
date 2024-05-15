#!/bin/bash
#
# Uses grobid to parse directory of pdf files. Requires a running grobid server.
# Instructions for setting up a local server using Docker can be found here:
# https://grobid.readthedocs.io/en/latest/Grobid-docker/. By default, the server 
# is assumed to be running on the address: http://localhost:8070. The address 
# can be modified in a config.json file. 
#
# To run this file, the grobid python client must also be installed: 
# https://github.com/kermitt2/grobid_client_python.

BASE_DIR="/Users/kkumbier/github/persisters/papers/Persisters_MRD_2024-05-09/"
PDF_DIR="$BASE_DIR/pdf"
XML_DIR="$BASE_DIR/xml"
CONFIG="./config.json"

grobid_client --input $PDF_DIR \
  --output $XML_DIR \
  --verbose \
  --consolidate_header \
  --n 6 \
  processFulltextDocument

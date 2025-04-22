#!/bin/sh
if [ "$1" = "--shell" ]; then
  exec /bin/sh
else
  exec python3 tinyfabulist.py "$@"
fi

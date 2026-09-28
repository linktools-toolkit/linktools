#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "$0"  )" && pwd  )"

cd "$CURRENT_DIR" && ./gradlew :adbd:publishToMavenLocal :framework:publishToMavenLocal :plugin:publishToMavenLocal

#!/bin/bash

# report out any errors
set -e

# add the user and email to git config
git config --global --add user.email ${GIT_AUTHOR_EMAIL} && git config --global --add user.name ${GIT_AUTHOR_NAME}

# Install Claude CLI for the non-root user
curl -fsSL https://claude.ai/install.sh | bash
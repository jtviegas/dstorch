#!/usr/bin/env bash

# ===> HEADER SECTION START  ===>

# http://bash.cumulonim.biz/NullGlob.html
shopt -s nullglob
# -------------------------------
this_folder="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ -z "$this_folder" ]; then
  this_folder=$(dirname $(readlink -f $0))
fi
parent_folder=$(dirname "$this_folder")

# -------------------------------
# --- required functions
debug(){
    local __msg="$1"
    echo " [DEBUG] `date` ... $__msg "
}

info(){
    local __msg="$1"
    echo " [INFO]  `date` ->>> $__msg "
}

warn(){
    local __msg="$1"
    echo " [WARN]  `date` *** $__msg "
}

err(){
    local __msg="$1"
    echo " [ERR]   `date` !!! $__msg "
}

file_age_days() {
  local file="$1"
  local file_time
  local current_time

  if [[ "$OSTYPE" == "darwin"* ]]; then
      file_time=$(stat -f %m "$file")
  else
      file_time=$(stat -c %Y "$file")
  fi

  current_time=$(date +%s)
  echo $(( (current_time - file_time) / 86400 ))
}

# ---------- CONSTANTS ----------
export FILE_VARIABLES=${FILE_VARIABLES:-".variables"}
export FILE_LOCAL_VARIABLES=${FILE_LOCAL_VARIABLES:-".local_variables"}
export FILE_SECRETS=${FILE_SECRETS:-".secrets"}
export INCLUDE_FILE=".bashutils"

# -------------------------------
# --- source variables files
if [ ! -f "$this_folder/$FILE_VARIABLES" ]; then
  warn "we DON'T have a $FILE_VARIABLES variables file - creating it"
  touch "$this_folder/$FILE_VARIABLES"
else
  . "$this_folder/$FILE_VARIABLES"
fi

if [ ! -f "$this_folder/$FILE_LOCAL_VARIABLES" ]; then
  warn "we DON'T have a $FILE_LOCAL_VARIABLES variables file - creating it"
  touch "$this_folder/$FILE_LOCAL_VARIABLES"
else
  . "$this_folder/$FILE_LOCAL_VARIABLES"
fi

if [ ! -f "$this_folder/$FILE_SECRETS" ]; then
  warn "we DON'T have a $FILE_SECRETS secrets file - creating it"
  touch "$this_folder/$FILE_SECRETS"
else
  . "$this_folder/$FILE_SECRETS"
fi

# ---------- include bashutils ----------
# --- refresh file if older than 1 day
bashutils="$this_folder/$INCLUDE_FILE"
[ $(file_age_days "$bashutils") -gt 1 ] && \
  curl -sf https://raw.githubusercontent.com/jtviegas/bashutils/master/.bashutils -o "${bashutils}.tmp" && \
  mv "${bashutils}.tmp" "$bashutils"
# --- source it
. $bashutils

# <=== HEADER SECTION END  <===


# ===> MAIN SECTION    ===>
# ---------- CONSTANTS ----------
export SRC_DIR=${SRC_DIR:-"${this_folder}/src"}
export TEST_DIR=${TEST_DIR:-"${this_folder}/tests"}
# -------------------------------
# --- main functions
install_qa_libs(){
  info "[install_qa_libs|in]"
  _pwd=`pwd`
  cd "$this_folder"

  pip install bandit==1.8.3 safety==3.5.1 "typer<0.17.0"
  local result="$?"
  if [ ! "$result" -eq "0" ] ; then err "[install_qa_libs] could not install dependencies"; fi

  cd "$_pwd"
  local msg="[install_qa_libs|out] => ${result}"
  [[ ! "$result" -eq "0" ]] && info "$msg" && exit 1
  info "$msg"
}

uninstall_qa_libs(){
  info "[uninstall_qa_libs|in]"
  _pwd=`pwd`
  cd "$this_folder"

  pip uninstall -y bandit safety typer
  local result="$?"
  if [ ! "$result" -eq "0" ] ; then err "[uninstall_qa_libs] could not install dependencies"; fi

  cd "$_pwd"
  local msg="[uninstall_qa_libs|out] => ${result}"
  [[ ! "$result" -eq "0" ]] && info "$msg" && exit 1
  info "$msg"
}

reqs(){
  info "[reqs|in]"
  _pwd=`pwd`
  cd "$this_folder"

  uv sync
  local result="$?"
  if [ ! "$result" -eq "0" ] ; then err "[reqs] could not install dependencies"; fi

  cd "$_pwd"

  local msg="[reqs|out] => ${result}"
  [[ ! "$result" -eq "0" ]] && info "$msg" && exit 1
  info "$msg"
}

unit_test(){
  info "[unit_test|in] ($1)"

  local FOLDER=$TEST_DIR
  [[ ! -z "$1" ]] && FOLDER="$1"

  _pwd=`pwd`
  cd "$this_folder"

  uv run pytest "$FOLDER" -x -s -vv --durations=0 \
    --cov="$SRC_DIR" \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --junitxml=unit-tests-results.xml
  local result="$?"
  [[ ! "$result" -eq "0" ]] && err "[unit_test] tests failed"
  cd "$_pwd"

  local msg="[unit_test|out] => ${result}"
  [[ ! "$result" -eq "0" ]] && info "$msg" && exit 1
  info "$msg"
}

unit_test_print_coverage()
{
  info "[unit_test_print_coverage|in]"
  
  uv run coverage report --show-missing
  uv run coverage html
  uv run coverage xml
  result="$?"
  [ "$result" -ne "0" ] && exit 1
  info "[unit_test_print_coverage|out] => $result"
  return ${result}
}

unit_test_coverage_check()
{
  info "[unit_test_coverage_check|in] ($1)"
  [ -z "$1" ] && usage

  local threshold=$1
  score=$(uv run coverage report | awk '$1 == "TOTAL" {print $NF+0}')
  result="$?"
  [ "$result" -ne "0" ] && exit 1
  if (( $threshold > $score )); then
    err "[unit_test_coverage_check] $score doesn't meet $threshold"
    exit 1
  fi
  info "[unit_test_coverage_check|out] => $score"
}

build(){
  info "[build|in]"

  _pwd=`pwd`
  cd "$this_folder"
  # changelog
  rm -rf dist/*
  uv build
  local result="$?"
  [[ ! "$result" -eq "0" ]] && err "[build] build failed"

  cd "$_pwd"
  local msg="[build|out] => ${result}"
  [[ ! "$result" -eq "0" ]] && info "$msg" && exit 1
  info "$msg"
}

publish(){
  info "[publish|in]"

  _pwd=`pwd`
  cd "$this_folder"

  uv publish --token "$PYPI_TOKEN"
  local result="$?"
  [[ ! "$result" -eq "0" ]] && err "[publish] publish failed"

  cd "$_pwd"
  local msg="[publish|out] => ${result}"
  [[ ! "$result" -eq "0" ]] && info "$msg" && exit 1
  info "$msg"
}

# <=== MAIN SECTION END  <===


# ===> FOOTER SECTION START  ===>

usage() {
  cat <<EOM
  usage:
  $(basename $0) { option }
    options:
      - install_qa_libs                  installs QA requirements (bandit, safety)
      - uninstall_qa_libs                uninstalls QA requirements (bandit, safety)
      - reqs                              installs development requirements
      - linter_check                      runs code lint and format check
      - sast_check                        runs static application security tests (SAST) check
      - sca_check                         runs software component analysis (SCA) check
      - test [<test_folder>]              runs unit tests
      - test_coverage                     prints test coverage report
      - test_coverage_check <threshold>   checks coverage against a threshold
      - build                             builds the package
      - publish                           publishes the package
      - tag <VERSION> <COMMIT_HASH>       tags a specific commit with the version and pushes it to the remote
      - get_latest_tag
EOM
  exit 1
}


case "$1" in
  install_qa_libs)
    install_qa_libs
    ;;
  uninstall_qa_libs)
    uninstall_qa_libs
    ;;
  reqs)
    reqs
    ;;
  linter_check)
    lint_check_ruff
    ;;
  sast_check)
    sast_check_bandit "$this_folder/src"
    ;;
  sca_check)
    sca_check_safety "$SAFETY_KEY"
    ;;
  test)
    unit_test "$2"
    ;;
  test_coverage)
    unit_test_print_coverage
    ;;
  test_coverage_check)
    unit_test_coverage_check "$2"
    ;;
  build)
    build
    ;;
  publish)
    publish
    ;;
  tag)
    git_tag_and_push "$2" "$3"
    ;;
  get_latest_tag)
    get_latest_tag
    ;;
  *)
    usage
    ;;
esac

# <=== FOOTER SECTION END  <===

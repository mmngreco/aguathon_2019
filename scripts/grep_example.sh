grep -lr "JUMP=72" ./ | xargs grep -l "SCORE_TEST=0.06" | xargs grep -r "COMMAND"
grep -r -l -E "SCORE_TEST=0\.0[1][1-2][0-2]" | xargs grep -r "SCORE_TEST"

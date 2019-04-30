grep -lr "JUMP=72" ./ | xargs grep -l "SCORE_TEST=0.06" | xargs grep -r "COMMAND"

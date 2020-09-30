This repository contains a python-based epidemic model.

It was used to study reopening Cornell University in this report:
https://people.orie.cornell.edu/pfrazier/COVID_19_Modeling_Jun15.pdf

It was also used to study national-level group screening protocols in this whitepaper:
https://docs.google.com/document/d/1joxMjHdWWo9XLFqfTdNXPQRAfeMjHYEyvVljqNCaKyE/

It is called "group-testing" because the original purpose was to study group testing protocols. 

FYI, when running via CHTC, you can connect to worker monitor by mapping port from the submit node to a local port using `ssh -L localhost:[local port]:localhost:[remote port] [username]@[submit node url]`. This allows you to access the monitor on your local machine at `localhost:[local port]` via a web browser (must have bokeh package installed). I'll typically do this after launching the process on the submit node, and allowing it to choose a port to host the monitor on (it will report it after launching the process).

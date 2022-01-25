# Contributing to the code

This guide introduces you to work and contribute to this repository.
The prerequisites are:
 - git: if it is not installed in your FBK machine, contact help-it;
 - PyCharm Professional Edition: students (including PhD) have free licence
for the Professional Edition of PyCharm using their institutional account (e.g. UniTN).
The Professional Edition has important features mentioned in this guide
that are not available in the Community (see Debugging on Cluster), so do use this version.


To start working on this code, first download the repository with `git clone`.
The master branch containing the up-to-date FBK MT Fairseq internal version is `internal_master`,
so you can access it entering into the cloned folder and running the command `git checkout internal_master`.
To install the repository, run `pip install -e .`.
We recommend installing the repository in a dedicated python virtual environment,
which you can create with PyCharm when importing the project or on command line.
In alternative, you can create a dedicated Anaconda environment.
We strongly discourage to install fairseq on your system Python version.

## Branching strategies

To start contributing to the project, first create a new branch from the updated `internal_master` branch, running:

```
git checkout -b im_new_feature
```

We follow the policy of naming the branch with the prefix `im_` to distinguish these branches
from branches coming from the external fairseq repository and/or old branches, or other test branches.

Once the code is ready, commit and push it upstream:

```
git add ...
git commit -m "your commit message"
git push -u origin im_new_feature
```

and create a new merge request (MR) to the `internal_master` branch.
The MR can be created from the gitlab website (see the "Merge Request" tab
in the project page).
When creating the MR, please follow these guidelines:

 1. Start the title of the MR with one or more tag in squared brackets identifying the topic of the work
(e.g. `[GENDER][LM]` for a work on language models [LM] for gender bias)
 2. From the dropdown menu in the `Description` section, choose `default` and then click on `Apply template`;
 3. Fill the template aswering the three questions that compose the three section of our template;
 4. If possible, assign a label that describes the project to which the work is related;
 5. Check that the CI (Continuous Integration, which runs automatically for each merge request) runs your tests and that it succedes;
 6. Suggest someone as reviewer of your MR.

Reviewers' comments and/or CI failures can be addressed by creating new commit(s) with
the needed changes and pushing them into the remote branch.
When the branch is ready, the reviewer will merge the changes into
the `internal_master` branch. The merge is handled by the script `fbk_dev/merge_mr.py`,
which ensures the proper formatting of the commit.
In case of merge conflicts with the `internal_master` branch (another commit conflicting with the current one
has been pushed), merge the `internal_master` branch into your feature branch and solve the conflicts.
Once the conflicts are solved, commit and push.

## Test Driven Development (TDD)

We encourage the adoption of a TDD method.
TDD is a simple paradigm which consists in FIRST writing unit tests (UTs);
SECOND asserting that the written tests are failing;
THIRD writing/fixing the code to make tests pass.
It is important to strictly follow this order to avoid
common errors, such as: writing a test that passes even before the patch
(it is usual when someone has not understood the root cause of an issue);
writing tests that check how the code actually works and not how it should work.


The rationale behind this paradigm draws from the idea that the code
should be thought for the users and not for the programmer.
Hence, writing first tests, i.e. code that uses the new features introduced
helps thinking to a better code structure.
Contextually, writing tests helps:

 1. Code understanding: you can check your assumptions on the reasons of a behavior,
you need to understand what each part of the code is responsible for
to write small, focused tests;
 2. Debugging: when writing code, you have an immediate method to test it and verify
the behavior. You can also debug your application in simple, controlled scenarios,
easy to understand.
 3. Documentation: tests are also a reference on how the code should be used 
and how it works.


Hence, although nobody can check whether you follow this paradigm or not,
if you care about the robustness, and quality of your code,
we strongly encourage its adoption.

Notice that reviewers (as well as authors) should enforce that all MRs (unless special cases
motivated by the excessive cost or impossibility of writing simple UTs)
contain UTs. UTs enforce correctness of the code both at the moment of MR submission
and in the future, ensuring that the introduced functionality is not broken in
future changes. For this reason, they are of the utmost importance to
collaborate with other people, benefiting from their work without the risk
of breaking changes.

# Debugging on cluster

Debugging on the cluster is a complex and costly approach that should be the
last resort to understand the behavior of the system, but can be useful/unavoidable
in complex cases related to specific data or conditions that cannot be replicated
otherwise.

Before you read this post, if you are new to the idea of remote
debugging, we advise to take a look at internet resources explaining it.
The procedure explained here assumes you use PyCharm as IDE. In case you
have a different IDE, most likely it will support similar remote
debugging but some configuration/steps may be different. I am going to
highlight the steps which are PyCharm specific, so you can replace them
with the similar procedure requested by your IDE.

First, if you use PyCharm, you need to setup the remote debug server.
Hence, please follow the instructions here:
https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.htmltag:remote-debug-config.
As remote server IP, you can set localhost. You can choose any port you
like: throughout the post, we'll name P1 this port you've chosen. Do
remember to properly fill the path mappings otherwise breakpoints will not
work. As you can see in the link, this involves also installing the
pycharm client on your cluster python.

Then, you need to put on the cluster frontend a copy of the java TCP forwarder that you can
download at http://jtcpfwd.sourceforge.net/. You also need to have java
on the cluster frontend.

Once all these prerequisites are set, the steps to do remote debugging on
the cluster nodes (workers) are:

 1. (If you use PyCharm) Add the 2 lines of code which you can see in the
Remote Debug configuration window to your code. The only things you need
to edit are: replace `localhost` with `frontend` and replace P1 (the port
you've configured in PyCharm) with another port you can freely choose:
we'll call it P2.
 2. On your local machine, create an SSH tunnel to the frontend, forwarding the
port you have chosen in the IDE, ie. P1. You can do this by running: `ssh
diclubclsin -R P1:localhost:P1 -N`. This will redirect all the traffic
incoming to `diclubclsin:P1` to your machine on P1 port.

 3. SSH in `diclubclsin` and start there the TCP forwarder from port P2 to port P1,
ie. run `java -jar jTCPfwd-lite.jar P2 localhost:P1`. This will redirect
all the traffic incoming to `diclubclsin:P2` to `diclubclsin:P1`.

 4. (In PyCharm) start your debugger, which will then wait for a
connection...

 5. Run your code in the workers by submitting it 
with `sbatch` or running it into your `srun` session.


You should have your remote debugging working now.

If you are unsure what's happening and you're curious, the network flow
is the following. The debugger client which runs in your code, tries to
connect to `diclubclsin:P2`. The TCP forwarder just forward everything to
`diclubclsin:P1`. The SSH tunnel, then, forwards from `diclubclsin:P1` to your local
machine on P1, where the debug server started by your IDE is running.

window.addEventListener("load", 
    function () {
        $("a.binder-button").map(function () {
            re = /(https:\/\/mybinder\.org.*\/)gh\/(.*\/)(.*)\/(.*)\?(urlpath=lab\/tree\/)(.*)/
            this.href = this.href.replace(re, function (match, p1, p2, p3, p4, p5, p6, offset, string) {
                p2_ = ['https://gitlab1.cs.cityu.edu.hk', '/', p2, p3].join('');
                return [p1, 'git/', encodeURIComponent(p2_), '/', p4, '?', 'urlpath=git-pull?', 
                    encodeURIComponent(['repo=', p2_, '&branch=', p4,'&', p5, p3, '/',  p6].join(''))].join('');
            })
        });
        $("a.jupyterhub-button").map(function () {
            re = /(https:\/\/.*)https:\/\/github\.com(\/.*)/
            this.href = this.href.replace(re, function (match, p1, p2, offset, string) {
                return [p1, 'https://gitlab1.cs.cityu.edu.hk', p2].join('');
            })
        });
        $("a.repository-button,a.issues-button,a.edit-button").map(function () {
            re = /https:\/\/github\.com(\/.*)/
            this.href = this.href.replace(re, function (match, p1, offset, string) {
                return ['https://gitlab1.cs.cityu.edu.hk', p1].join('');
            })
        });
        $(".dropdown-buttons-trigger i.fa-github").addClass('fa-gitlab').removeClass('fa-github');
    }
)


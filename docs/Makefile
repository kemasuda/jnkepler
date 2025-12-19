# docs/Makefile

SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build
APIDIR        = api
PKGDIR        = ../src/jnkepler

.PHONY: apidoc html clean

apidoc:
	rm -rf $(APIDIR)
	$(SPHINXAPIDOC) -f -o $(APIDIR) $(PKGDIR)

html: apidoc
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR) $(APIDIR)

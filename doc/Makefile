SPHINXBUILD ?= python -m sphinx
SOURCEDIR = source
BUILDDIR = build

.PHONY: html clean linkcheck

html:
	$(SPHINXBUILD) -W --keep-going -b html $(SOURCEDIR) $(BUILDDIR)/html

linkcheck:
	$(SPHINXBUILD) -W --keep-going -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

clean:
	$(RM) -r $(BUILDDIR)

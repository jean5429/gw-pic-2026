"""Conversor de um subconjunto de LaTeX para OMML (equacoes nativas do Word).

Suporta: \\frac, \\sqrt, ^ e _, \\left...\\right, \\int/\\sum/\\prod (o corpo vai ate o fim
do grupo atual; use chaves para limitar), \\tilde/\\hat/\\bar, \\mathrm/\\text/\\operatorname,
\\mathbf, \\boldsymbol, \\mathcal, letras gregas, simbolos usuais, \\max/\\min/\\argmax com
limite embaixo e espacos (\\, \\; \\quad \\qquad \\ ).
"""
from __future__ import annotations

import re

from lxml import etree

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "ϕ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "ell": "ℓ",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOLS = {
    "infty": "∞", "partial": "∂", "nabla": "∇", "cdot": "·", "times": "×", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "sim": "∼", "propto": "∝",
    "pm": "±", "to": "→", "rightarrow": "→", "leftarrow": "←", "in": "∈", "ldots": "…",
    "cdots": "⋯", "odot": "⊙", "top": "⊤", "ast": "∗", "prime": "′", "circ": "∘",
    "equiv": "≡", "langle": "⟨", "rangle": "⟩", "mid": "∣", "sun": "⊙",
}
# O Word ignora espacos comuns dentro de equacoes; por isso todos viram espacos tipograficos.
SPACES = {",": " ", ";": " ", ":": " ", "quad": " ", "qquad": "  ",
          "!": "", " ": " "}
FUNCS = {"ln", "log", "exp", "max", "min", "sin", "cos", "tan", "arg", "det", "lim", "sup", "inf",
         "argmax", "argmin"}
LIMOPS = {"max", "min", "lim", "sup", "inf", "argmax", "argmin"}
ACCENTS = {"tilde": "̃", "widetilde": "̃", "hat": "̂", "widehat": "̂",
           "vec": "⃗", "dot": "̇"}
NARY = {"\\int": "∫", "\\sum": "∑", "\\prod": "∏", "\\oint": "∮", "\\iint": "∬"}
CAL = {"M": "ℳ", "L": "ℒ", "N": "𝒩", "O": "𝒪", "H": "ℋ", "E": "ℰ", "F": "ℱ", "B": "ℬ", "R": "ℛ"}

TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|\d+(?:[.,]\d+)?|[{}^_\[\]]|\s+|.", re.S)


def m_el(tag: str, parent=None, **attrs):
    name = f"{{{M_NS}}}{tag}"
    element = etree.SubElement(parent, name) if parent is not None else etree.Element(name)
    for key, value in attrs.items():
        element.set(f"{{{M_NS}}}{key}", value)
    return element


def m_run(text: str, style: str | None = None, normal: bool = False):
    run = m_el("r")
    if style or normal:
        properties = m_el("rPr", run)
        if normal:
            # Texto comum dentro da equacao: hifens e espacos ficam como estao.
            m_el("nor", properties)
        else:
            m_el("sty", properties, val=style)
    word_properties = etree.SubElement(run, f"{{{W_NS}}}rPr")
    fonts = etree.SubElement(word_properties, f"{{{W_NS}}}rFonts")
    fonts.set(f"{{{W_NS}}}ascii", "Cambria Math")
    fonts.set(f"{{{W_NS}}}hAnsi", "Cambria Math")
    text_element = m_el("t", run)
    text_element.text = text
    text_element.set(XML_SPACE, "preserve")
    return run


def _fill(parent, children) -> None:
    for child in children:
        parent.append(child)


class Tex:
    def __init__(self, source: str):
        self.source = source
        self.tokens = TOKEN_RE.findall(source)
        self.i = 0

    # -- leitura de tokens
    def peek(self):
        j = self.i
        while j < len(self.tokens) and self.tokens[j].isspace():
            j += 1
        return self.tokens[j] if j < len(self.tokens) else None

    def take(self):
        while self.i < len(self.tokens) and self.tokens[self.i].isspace():
            self.i += 1
        if self.i >= len(self.tokens):
            raise ValueError(f"fim inesperado em: {self.source}")
        token = self.tokens[self.i]
        self.i += 1
        return token

    def expect(self, token: str) -> None:
        got = self.take()
        if got != token:
            raise ValueError(f"esperado {token!r}, veio {got!r} em: {self.source}")

    # -- gramatica
    def parse(self):
        items = self.expr(None)
        if self.peek() is not None:
            raise ValueError(f"sobrou {self.peek()!r} em: {self.source}")
        return items

    def expr(self, stop):
        items = []
        while True:
            token = self.peek()
            if token is None:
                if stop is not None:
                    raise ValueError(f"faltou {stop!r} em: {self.source}")
                return items
            if token == stop:
                return items
            if token in NARY:
                items.append(self.nary(stop))
                continue
            items.extend(self.scripts(self.atom()))

    def arg(self):
        if self.peek() == "{":
            self.take()
            inner = self.expr("}")
            self.expect("}")
            return inner
        return self.atom()

    def raw_arg(self) -> str:
        self.expect("{")
        depth, parts = 1, []
        while True:
            token = self.tokens[self.i]
            self.i += 1
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
                if depth == 0:
                    return "".join(parts)
            parts.append(token)

    def delimiter(self) -> str:
        token = self.take()
        mapping = {".": "", "\\{": "{", "\\}": "}", "\\lbrace": "{", "\\rbrace": "}", "\\|": "‖",
                   "\\Vert": "‖", "\\langle": "⟨", "\\rangle": "⟩", "\\lvert": "|", "\\rvert": "|"}
        return mapping.get(token, token)

    def scripts(self, base):
        sub = sup = None
        while self.peek() in ("^", "_"):
            token = self.take()
            value = self.arg()
            if token == "^":
                sup = value
            else:
                sub = value
        if sub is None and sup is None:
            return base
        if sub is not None and sup is not None:
            element = m_el("sSubSup")
            _fill(m_el("e", element), base)
            _fill(m_el("sub", element), sub)
            _fill(m_el("sup", element), sup)
        elif sup is not None:
            element = m_el("sSup")
            _fill(m_el("e", element), base)
            _fill(m_el("sup", element), sup)
        else:
            element = m_el("sSub")
            _fill(m_el("e", element), base)
            _fill(m_el("sub", element), sub)
        return [element]

    def nary(self, stop):
        token = self.take()
        sub = sup = None
        while self.peek() in ("^", "_"):
            script = self.take()
            value = self.arg()
            if script == "^":
                sup = value
            else:
                sub = value
        body = self.expr(stop)
        element = m_el("nary")
        properties = m_el("naryPr", element)
        m_el("chr", properties, val=NARY[token])
        m_el("limLoc", properties, val="subSup" if token in ("\\int", "\\oint", "\\iint") else "undOvr")
        if sub is None:
            m_el("subHide", properties, val="1")
        if sup is None:
            m_el("supHide", properties, val="1")
        _fill(m_el("sub", element), sub or [])
        _fill(m_el("sup", element), sup or [])
        _fill(m_el("e", element), body)
        return element

    def atom(self):
        token = self.take()
        if token == "{":
            inner = self.expr("}")
            self.expect("}")
            return inner
        if token.startswith("\\") and len(token) > 1:
            return self.command(token[1:])
        if token in ("^", "_", "}"):
            raise ValueError(f"token fora de lugar {token!r} em: {self.source}")
        if token == "-":
            return [m_run("−")]
        if token == "*":
            return [m_run("∗")]
        if token == "'":
            return [m_run("′")]
        return [m_run(token)]

    def command(self, name: str):
        if name in ("frac", "dfrac", "tfrac"):
            numerator, denominator = self.arg(), self.arg()
            element = m_el("f")
            _fill(m_el("num", element), numerator)
            _fill(m_el("den", element), denominator)
            return [element]
        if name == "sqrt":
            degree = None
            if self.peek() == "[":
                self.take()
                degree = self.expr("]")
                self.expect("]")
            body = self.arg()
            element = m_el("rad")
            properties = m_el("radPr", element)
            if degree is None:
                m_el("degHide", properties, val="1")
            _fill(m_el("deg", element), degree or [])
            _fill(m_el("e", element), body)
            return [element]
        if name == "left":
            begin = self.delimiter()
            inner = self.expr("\\right")
            self.expect("\\right")
            end = self.delimiter()
            element = m_el("d")
            properties = m_el("dPr", element)
            m_el("begChr", properties, val=begin)
            m_el("endChr", properties, val=end)
            _fill(m_el("e", element), inner)
            return [element]
        if name in ACCENTS:
            body = self.arg()
            element = m_el("acc")
            properties = m_el("accPr", element)
            m_el("chr", properties, val=ACCENTS[name])
            _fill(m_el("e", element), body)
            return [element]
        if name in ("bar", "overline"):
            body = self.arg()
            element = m_el("bar")
            properties = m_el("barPr", element)
            m_el("pos", properties, val="top")
            _fill(m_el("e", element), body)
            return [element]
        if name in ("text", "textrm"):
            return [m_run(self.raw_arg(), normal=True)]
        if name in ("mathrm", "operatorname"):
            return [m_run(self.raw_arg(), "p")]
        if name == "mathbf":
            return [m_run(self.raw_arg(), "b")]
        if name == "boldsymbol":
            text = self.raw_arg().strip()
            if text.startswith("\\") and text[1:] in GREEK:
                letter = GREEK[text[1:]]
                return [m_run(letter, "bi" if text[1].islower() else "b")]
            return [m_run(text, "bi")]
        if name == "mathcal":
            return [m_run("".join(CAL.get(ch, ch) for ch in self.raw_arg()), "p")]
        if name in FUNCS:
            word = {"argmax": "arg max", "argmin": "arg min"}.get(name, name)
            run = m_run(word, "p")
            if name in LIMOPS and self.peek() == "_":
                self.take()
                limit = self.arg()
                element = m_el("limLow")
                _fill(m_el("e", element), [run])
                _fill(m_el("lim", element), limit)
                return [element]
            return [run]
        if name in GREEK:
            return [m_run(GREEK[name], None if name[0].islower() else "p")]
        if name in SYMBOLS:
            return [m_run(SYMBOLS[name], "p")]
        if name in SPACES:
            return [m_run(SPACES[name], "p")] if SPACES[name] else []
        if len(name) == 1 and not name.isalpha():
            return [m_run({"|": "‖"}.get(name, name), "p")]
        raise ValueError(f"comando desconhecido \\{name} em: {self.source}")


def latex_to_omml(source: str):
    """Lista de elementos OMML para colocar dentro de um m:oMath."""
    return Tex(source).parse()


def omath(source: str):
    element = m_el("oMath")
    _fill(element, latex_to_omml(source))
    return element


def omath_para(source: str):
    paragraph = m_el("oMathPara")
    properties = m_el("oMathParaPr", paragraph)
    m_el("jc", properties, val="center")
    paragraph.append(omath(source))
    return paragraph


def set_math_size(element, size_pt: float) -> None:
    """Fixa o tamanho de todas as letras de uma equacao (w:sz em meios-pontos)."""
    half_points = str(int(round(size_pt * 2)))
    for properties in element.iter(f"{{{W_NS}}}rPr"):
        for tag in ("sz", "szCs"):
            node = properties.find(f"{{{W_NS}}}{tag}")
            if node is None:
                node = etree.SubElement(properties, f"{{{W_NS}}}{tag}")
            node.set(f"{{{W_NS}}}val", half_points)

import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Link, useNavigate, useLocation, useParams } from "react-router-dom";

// CQ2 Atlas – Marketing Site + Dashboard Shell (Soft Green Finance Theme)
// Enriched Home inspired by premium climate/ESG sites (hero imagery, use cases, trust badges,
// resources, case studies, press, newsletter) — all placeholders you can swap with real assets.

// ===== Design Tokens =====
const COLORS = {
  bg: "#FAFDF8",
  card: "#FFFFFF",
  ink: "#202B21",
  sub: "#5B6D5F",
  border: "#E5F0E0",
  mint: "#E8F6E1",
  mintDeep: "#CDE8C5",
  pill: "#CBE7B9",
  yellow: "#FBE79B",
  yellowDeep: "#F6D865",
  danger: "#F4B4B4",
} as const;

// ===== Asset Paths (drop your files in /public/assets/ ) =====
const ASSETS = {
  logo: "/assets/cq2-logo.png",
  hero: "/assets/hero-carbon-grid.jpg",
  map: "/assets/contagion-map.png",
  policy: "/assets/policy-docs.jpg",
  optimizer: "/assets/optimizer-cards.jpg",
  customer1: "/assets/logo-cust-1.svg",
  customer2: "/assets/logo-cust-2.svg",
  customer3: "/assets/logo-cust-3.svg",
  customer4: "/assets/logo-cust-4.svg",
  press1: "/assets/press-1.png",
  press2: "/assets/press-2.png",
  press3: "/assets/press-3.png",
  // Team photos — add your real files here (prefer PNG/SVG or JPG 600x600)
  team1: "/assets/team-tan-chun-loong.jpg",
  team2: "/assets/team-tan-qian-wen.jpg",
  team3: "/assets/team-tee-hui-en.jpg",
  team4: "/assets/team-lim-en-wei.jpg",
  team5: "/assets/team-x-placeholder.jpg",
};

// ===== Primitives =====
const Container: React.FC<React.PropsWithChildren<{ className?: string; style?: React.CSSProperties }>> = ({ className = "", style, children }) => (
  <div className={`mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 ${className}`} style={style}>{children}</div>
);

const Pill: React.FC<React.PropsWithChildren<{ tone?: "mint"|"ink"|"yellow"; className?: string; style?: React.CSSProperties }>> = (props) => {
  const { tone = "mint", className = "", children, style } = props as any;
  const bg = tone === "mint" ? COLORS.pill : tone === "yellow" ? COLORS.yellow : COLORS.ink;
  const fg = tone === "ink" ? "#FFFFFF" : COLORS.ink;
  return <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ${className}`} style={{ background: bg, color: fg, ...style }}>{children}</span>;
};

const Button: React.FC<React.PropsWithChildren<{ variant?: "primary"|"ghost"|"link"; to?: string; className?: string; onClick?: () => void; style?: React.CSSProperties }>> = (props) => {
  const { variant = "primary", to, className = "", onClick, children, style } = props as any;
  const El: any = to ? Link : "button";
  const base = "inline-flex items-center justify-center rounded-full px-5 py-2.5 text-sm font-medium transition";
  const styles = variant === "primary"
    ? { background: COLORS.ink, color: "#fff" }
    : variant === "ghost"
      ? { background: COLORS.mint, color: COLORS.ink, border: `1px solid ${COLORS.border}` }
      : { background: "transparent", color: COLORS.ink };
  return <El to={to as any} onClick={onClick} className={`${base} ${className}`} style={{ ...styles, ...style }}>{children}</El>;
};

const Card: React.FC<React.PropsWithChildren<{ title?: string; right?: React.ReactNode; className?: string }>> = ({ title, right, className = "", children }) => (
  <div className={`rounded-3xl border shadow-sm ${className}`} style={{ background: COLORS.card, borderColor: COLORS.border }}>
    {(title || right) && (
      <div className="px-6 py-4 border-b" style={{ borderColor: COLORS.border }}>
        <div className="flex items-center justify-between">
          <h3 className="text-[15px] font-semibold" style={{ color: COLORS.ink }}>{title}</h3>
          {right}
        </div>
      </div>
    )}
    <div className="p-6">{children}</div>
  </div>
);

// ===== Brand header / nav with smooth-scroll =====
const SiteHeader: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const goto = (id: string) => {
    const scroll = () => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (location.pathname !== '/') { navigate('/'); setTimeout(scroll, 60); } else { scroll(); }
  };
  return (
    <header className="border-b sticky top-0 z-30 backdrop-blur" style={{ background: "rgba(255,255,255,0.7)", borderColor: COLORS.border }}>
      <Container className="h-20 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <img src={ASSETS.logo} alt="CQ2 Atlas" className="h-9 w-9 rounded-xl object-contain" onError={(e:any)=>{ e.currentTarget.style.display='none'; }} />
          <div>
            <div className="text-base font-semibold" style={{ color: COLORS.ink }}>CQ2 Atlas</div>
            <div className="text-[11px]" style={{ color: COLORS.sub }}>Contagion‑Aware Carbon Risk</div>
          </div>
        </Link>
        <nav className="flex items-center gap-2">
          <button className="px-4 py-2 rounded-full text-sm hover:bg-black/5" style={{ color: COLORS.ink }} onClick={()=>goto('hero')}>Home</button>
          <button className="px-4 py-2 rounded-full text-sm hover:bg-black/5" style={{ color: COLORS.ink }} onClick={()=>goto('resources')}>Resources</button>
          <Link className="px-4 py-2 rounded-full text-sm hover:bg-black/5" style={{ color: COLORS.ink }} to="/team">Team</Link>
          <Link className="px-4 py-2 rounded-full text-sm hover:bg-black/5" style={{ color: COLORS.ink }} to="/dashboard">Dashboard</Link>
          <Button variant="ghost" className="ml-2">Log in</Button>
          <Button to="/dashboard" className="ml-1">Get started</Button>
        </nav>
      </Container>
    </header>
  );
};

const SiteFooter: React.FC = () => (
  <footer className="mt-16 border-t" style={{ borderColor: COLORS.border }}>
    <Container className="py-10 text-sm flex flex-col md:flex-row items-center justify-between gap-4" style={{ color: COLORS.sub }}>
      <div>© 2025 CQ2 Atlas — All rights reserved.</div>
      <div className="flex items-center gap-4">
        <Link to="/about" className="hover:underline">About</Link>
        <a href="#solutions" className="hover:underline" onClick={(e)=>{ e.preventDefault(); document.getElementById('solutions')?.scrollIntoView({behavior:'smooth'}); }}>Solutions</a>
        <Link to="/team" className="hover:underline">Team</Link>
        <Link to="/dashboard" className="hover:underline">Dashboard</Link>
      </div>
    </Container>
  </footer>
);

// ===== Pages =====
const HomePage: React.FC = () => (
  <div style={{ background: COLORS.bg, color: COLORS.ink }}>
    {/* Hero with imagery */}
    <section className="pt-16 md:pt-24" id="hero">
      <Container className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
        <div>
          <Pill>FINANCE · ESG · QUANTUM</Pill>
          <h1 className="mt-4 text-4xl md:text-6xl font-semibold leading-tight tracking-tight">
            Streamlined, <span className="rounded-[18px] px-2" style={{ background: COLORS.pill }}>Contagion‑aware</span> Carbon Risk Intelligence.
          </h1>
          <p className="mt-4 text-base md:text-lg" style={{ color: COLORS.sub }}>
            <b>CQ2 Atlas</b> maps policy contagion across countries, registries and project types to reveal systemic risk in your carbon portfolios.
          </p>
          <div className="mt-6 flex gap-3">
            <Button to="/dashboard">Open Dashboard</Button>
            <Button variant="ghost" to="/about">Learn more</Button>
          </div>
          <div className="mt-6 flex items-center gap-3 text-xs" style={{ color: COLORS.sub }}>
            <span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: COLORS.mintDeep }}></span>Compliance‑ready</span>
            <span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: COLORS.mintDeep }}></span>Audit‑grade reporting</span>
            <span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: COLORS.mintDeep }}></span>Quantum‑enabled</span>
          </div>
        </div>
        <div>
          <div className="rounded-3xl overflow-hidden border" style={{ borderColor: COLORS.border }}>
            <img src={ASSETS.hero} alt="Carbon grids" className="w-full h-[360px] object-cover" onError={(e:any)=>{ e.currentTarget.replaceWith(Object.assign(document.createElement('div'),{className:'h-[360px]',style:`background:${COLORS.mint}`})) }} />
          </div>
          <div className="grid grid-cols-3 gap-3 mt-3">
            <img src={ASSETS.map} alt="Contagion map" className="rounded-2xl border h-24 w-full object-cover" style={{ borderColor: COLORS.border }} onError={(e:any)=>{ e.currentTarget.style.display='none'; }} />
            <img src={ASSETS.policy} alt="Policy docs" className="rounded-2xl border h-24 w-full object-cover" style={{ borderColor: COLORS.border }} onError={(e:any)=>{ e.currentTarget.style.display='none'; }} />
            <img src={ASSETS.optimizer} alt="Optimizer" className="rounded-2xl border h-24 w-full object-cover" style={{ borderColor: COLORS.border }} onError={(e:any)=>{ e.currentTarget.style.display='none'; }} />
          </div>
        </div>
      </Container>
    </section>

    {/* Impact */}
    <section className="mt-20" id="impact">
      <Container>
        <div className="text-center max-w-3xl mx-auto">
          <Pill>Why it matters</Pill>
          <h2 className="text-3xl md:text-4xl font-semibold mt-2">Policy shocks propagate like contagion</h2>
          <p className="mt-3 text-sm md:text-base" style={{ color: COLORS.sub }}>
            Host‑country reversals, registry scandals, and Article 6 changes can cascade across interconnected markets. Atlas quantifies and visualizes these network effects so risk teams can act early.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-10">
          <Card title="Host‑country reversals"><p className="text-sm" style={{ color: COLORS.sub }}>Detect early warning signals from policy discourse and regulatory calendars.</p></Card>
          <Card title="Registry incidents"><p className="text-sm" style={{ color: COLORS.sub }}>Stress test cross‑registry contagion to avoid stranded or ineligible credits.</p></Card>
          <Card title="Article 6 rule changes"><p className="text-sm" style={{ color: COLORS.sub }}>Model bilateral agreements and ITMO dynamics into your portfolio risk.</p></Card>
        </div>
      </Container>
    </section>

    {/* Solutions */}
    <section className="mt-16" id="solutions">
      <Container>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card title="Contagion Graph">
            <p className="text-sm" style={{ color: COLORS.sub }}>Multi‑layer map of countries, registries, project types and sellers with configurable edge weights.</p>
          </Card>
          <Card title="Tail‑Risk Simulator">
            <p className="text-sm" style={{ color: COLORS.sub }}>Classical Monte Carlo today; Quantum Amplitude Estimation option for 95/99% tails.</p>
          </Card>
          <Card title="Portfolio Optimizer">
            <p className="text-sm" style={{ color: COLORS.sub }}>QUBO selection balances risk tiers, costs and Singapore’s 5% offset cap.</p>
          </Card>
        </div>
      </Container>
    </section>

    {/* Use Cases (like verticals) */}
    <section className="mt-16">
      <Container>
        <div className="text-center max-w-2xl mx-auto">
          <Pill>Use cases</Pill>
          <h3 className="text-2xl md:text-3xl font-semibold mt-2">Built for regulated emitters & market participants</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
          <Card title="CFO & Finance"><p className="text-sm" style={{ color: COLORS.sub }}>Tail‑loss budgeting, stranded‑credit avoidance, audit evidence.</p></Card>
          <Card title="ESG & Compliance"><p className="text-sm" style={{ color: COLORS.sub }}>Cap tracking, methodology governance, registry audit packs.</p></Card>
          <Card title="Procurement"><p className="text-sm" style={{ color: COLORS.sub }}>Vendor diversification, credit substitution playbooks.</p></Card>
          <Card title="Trading"><p className="text-sm" style={{ color: COLORS.sub }}>Scenario stress tests, cross‑registry spread monitoring.</p></Card>
        </div>
      </Container>
    </section>

    {/* Trust badges / Logos */}
    <section className="mt-16">
      <Container>
        <Card title="Trusted by">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 items-center opacity-80">
            {[ASSETS.customer1, ASSETS.customer2, ASSETS.customer3, ASSETS.customer4].map((src, i) => (
              <img key={i} src={src} alt={`Logo ${i+1}`} className="h-10 object-contain" onError={(e:any)=>{ e.currentTarget.replaceWith(Object.assign(document.createElement('div'),{className:'h-10 rounded-xl',style:`background:${COLORS.mint}`})) }} />
            ))}
          </div>
        </Card>
      </Container>
    </section>

    {/* Resources (like Sylvera) */}
    <section className="mt-16" id="resources">
      <Container>
        <div className="text-center max-w-2xl mx-auto">
          <Pill>Resources</Pill>
          <h3 className="text-2xl md:text-3xl font-semibold mt-2">Guides, whitepapers & commentary</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <Card title="Detecting Policy Shock Signals"><p className="text-sm" style={{ color: COLORS.sub }}>How to use headline embeddings and curated indices to spot early warnings.</p><Button variant="ghost">Read guide</Button></Card>
          <Card title="Registry Quality & Reliability"><p className="text-sm" style={{ color: COLORS.sub }}>A framework to translate registry incidents into contagion weights.</p><Button variant="ghost">Download PDF</Button></Card>
          <Card title="Quantum Tail‑Risk 101"><p className="text-sm" style={{ color: COLORS.sub }}>When QAE beats MC for extreme‑tail accuracy and speed.</p><Button variant="ghost">Learn more</Button></Card>
        </div>
      </Container>
    </section>

    {/* Press */}
    <section className="mt-16">
      <Container>
        <Card title="As seen in">
          <div className="grid grid-cols-3 gap-6 items-center">
            {[ASSETS.press1, ASSETS.press2, ASSETS.press3].map((src, i) => (
              <img key={i} src={src} alt={`Press ${i+1}`} className="h-8 object-contain opacity-70" onError={(e:any)=>{ e.currentTarget.style.display='none'; }} />
            ))}
          </div>
        </Card>
      </Container>
    </section>

    {/* CTA */}
    <section className="mt-16 mb-12">
      <Container>
        <div className="rounded-3xl p-8 flex flex-col md:flex-row items-center justify-between gap-4" style={{ background: COLORS.yellow }}>
          <div>
            <div className="text-xl font-semibold">Ready to see how your carbon portfolio weathers a policy shock?</div>
            <div className="text-sm mt-1" style={{ color: COLORS.sub }}>Open the dashboard and explore live risk tiers S/A/B/C/F.</div>
          </div>
          <div className="flex gap-3">
            <Button to="/dashboard">Get started</Button>
            <Button variant="ghost" to="/about">About CQ2 Atlas</Button>
          </div>
        </div>
      </Container>
    </section>

    {/* Newsletter */}
    <section className="mb-16">
      <Container>
        <Card title="Subscribe for policy contagion insights">
          <form className="flex flex-col md:flex-row gap-3">
            <input required type="email" placeholder="Work email" className="flex-1 rounded-full px-4 py-3 text-sm border" style={{ background: COLORS.card, borderColor: COLORS.border }} />
            <Button>Subscribe</Button>
          </form>
          <p className="text-xs mt-2" style={{ color: COLORS.sub }}>By subscribing you agree to our privacy policy.</p>
        </Card>
      </Container>
    </section>
  </div>
);

const AboutPage: React.FC = () => (
  <div style={{ background: COLORS.bg }}>
    <Container className="py-16">
      <div className="text-center max-w-3xl mx-auto">
        <Pill>About CQ2 Atlas</Pill>
        <h2 className="text-3xl md:text-4xl font-semibold mt-2" style={{ color: COLORS.ink }}>Clarity for policy‑driven carbon markets</h2>
        <p className="mt-3 text-sm md:text-base" style={{ color: COLORS.sub }}>We are an interdisciplinary team across quant finance, climate policy, and systems engineering. Our goal is to surface networked fragility before it becomes loss.</p>
      </div>
      <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Mission"><p className="text-sm" style={{ color: COLORS.sub }}>Enable regulated emitters and registries to anticipate cross‑market contagion and avoid stranded credits while complying with evolving standards.</p></Card>
        <Card title="Approach"><ul className="list-disc pl-5 text-sm" style={{ color: COLORS.sub }}><li>Policy shock detection → embeddings + curated policy index</li><li>Contagion propagation → multi‑layer Markov graph</li><li>Tail‑risk estimation → classical MC, quantum‑ready</li><li>Optimization → QUBO portfolio selection</li></ul></Card>
      </div>
      <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card title="Leadership"><ul className="text-sm space-y-2" style={{ color: COLORS.sub }}><li><b style={{ color: COLORS.ink }}>Founder, Quant Lead</b> — portfolio risk & Monte Carlo</li><li><b style={{ color: COLORS.ink }}>Co‑founder, Policy Lead</b> — Article 6 & registry governance</li><li><b style={{ color: COLORS.ink }}>Co‑founder, Systems Lead</b> — cloud & data engineering</li></ul></Card>
        <Card title="Advisors & Partners"><p className="text-sm" style={{ color: COLORS.sub }}>We collaborate with registries, emitters and research groups to validate contagion assumptions and stress tests.</p></Card>
        <Card title="Careers"><p className="text-sm" style={{ color: COLORS.sub }}>We’re building in Singapore & Southeast Asia. If you work at the intersection of climate and risk modelling, let’s talk.</p></Card>
      </div>
    </Container>
  </div>
);

// ===== Team Page =====
const TeamPage: React.FC = () => (
  <div style={{ background: COLORS.bg }}>
    <Container className="py-16">
      <div className="text-center max-w-3xl mx-auto">
        <Pill>Our Team</Pill>
        <h2 className="text-3xl md:text-4xl font-semibold mt-2" style={{ color: COLORS.ink }}>People behind CQ2 Atlas</h2>
        <p className="mt-3 text-sm md:text-base" style={{ color: COLORS.sub }}>Interdisciplinary builders across UI/UX, policy, quantum & classical engineering, and analytics.</p>
      </div>
      <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          {img: ASSETS.team1, name: 'Tan Chun Loong', title: 'Founder — UI/UX/Architecture'},
          {img: ASSETS.team2, name: 'Tan Qian Wen', title: 'Co‑Founder — Business Analyst'},
          {img: ASSETS.team3, name: 'Tee Hui En', title: 'Co‑Founder — Quantum Engineer'},
          {img: ASSETS.team4, name: 'Lim En Wei', title: 'Co‑Founder — Classical Engineer / Data Analyst'},
          {img: ASSETS.team5, name: 'XXXX XXXX', title: 'XXXX'},
        ].map((m, i) => (
          <Card key={i}>
            <div className="flex items-center gap-4">
              <div className="h-20 w-20 rounded-2xl overflow-hidden border" style={{ borderColor: COLORS.border }}>
                <img src={m.img} alt={m.name} className="h-full w-full object-cover" onError={(e:any)=>{ e.currentTarget.replaceWith(Object.assign(document.createElement('div'),{className:'h-full w-full',style:`background:${COLORS.mint}`})) }} />
              </div>
              <div>
                <div className="text-base font-semibold" style={{ color: COLORS.ink }}>{m.name}</div>
                <div className="text-sm" style={{ color: COLORS.sub }}>{m.title}</div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </Container>
  </div>
);

// ===== Dashboard Pages (refocused for Singapore green tech credit risk) =====

type CreditRating = 'Prime'|'Strong'|'Moderate'|'Watch'|'Stressed';

const RATING_STYLES: Record<CreditRating, { label: string; bg: string; fg: string }> = {
  Prime: { label: 'Prime', bg: COLORS.mint, fg: COLORS.ink },
  Strong: { label: 'Strong', bg: COLORS.mintDeep, fg: COLORS.ink },
  Moderate: { label: 'Moderate', bg: COLORS.yellow, fg: COLORS.ink },
  Watch: { label: 'Watch', bg: '#FFE2B8', fg: COLORS.ink },
  Stressed: { label: 'Stressed', bg: COLORS.danger, fg: '#7A1C1C' },
};

const RatingBadge: React.FC<{ rating: CreditRating }> = ({ rating }) => {
  const { label, bg, fg } = RATING_STYLES[rating];
  return (
    <span className="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold" style={{ background: bg, color: fg, border: `1px solid ${COLORS.border}` }}>
      {label}
    </span>
  );
};

interface ExposurePoint { day: string; pfe: number; ee: number; }
interface SentimentSnapshot { mean: number; change: number; summary: string; drivers: string[]; }

interface CompanyProfile {
  id: string;
  name: string;
  segment: string;
  location: string;
  rating: CreditRating;
  portfolioShare: number;
  limitUtilisation: number;
  exposures: ExposurePoint[];
  sentiment: SentimentSnapshot;
  liquidity: 'High'|'Adequate'|'Tight';
  analystNote: string;
  counterparties: string[];
}

const COMPANIES: CompanyProfile[] = [
  {
    id: 'SG-GTC-001',
    name: 'SolaraGrid Pte Ltd',
    segment: 'Utility-scale Solar',
    location: 'Singapore · Tuas',
    rating: 'Prime',
    portfolioShare: 0.28,
    limitUtilisation: 0.52,
    exposures: [
      { day: 'Day -5', pfe: 10.2, ee: 5.8 },
      { day: 'Day -4', pfe: 10.5, ee: 6.1 },
      { day: 'Day -3', pfe: 10.1, ee: 5.9 },
      { day: 'Day -2', pfe: 9.8, ee: 5.6 },
      { day: 'Day -1', pfe: 9.4, ee: 5.3 },
      { day: 'Day 0', pfe: 9.1, ee: 5.1 },
      { day: 'Day +1', pfe: 9.6, ee: 5.4 },
    ],
    sentiment: {
      mean: 0.62,
      change: 0.07,
      summary: 'Positive press on new 25MW PPAs with Keppel and stable REC pricing lifted sentiment.',
      drivers: ['Keppel supply agreement', 'URA rooftop tender win', 'Stable REC spreads'],
    },
    liquidity: 'High',
    analystNote: 'MAS green loan covenants met; collateral coverage improved after Q2 asset revaluation.',
    counterparties: ['DBS Project Finance', 'Keppel Utilities', 'SP Group'],
  },
  {
    id: 'SG-GTC-004',
    name: 'HarborWind Renewables',
    segment: 'Offshore Wind OEM',
    location: 'Singapore · Jurong',
    rating: 'Strong',
    portfolioShare: 0.19,
    limitUtilisation: 0.63,
    exposures: [
      { day: 'Day -5', pfe: 13.4, ee: 8.1 },
      { day: 'Day -4', pfe: 13.0, ee: 7.8 },
      { day: 'Day -3', pfe: 12.6, ee: 7.4 },
      { day: 'Day -2', pfe: 12.1, ee: 7.0 },
      { day: 'Day -1', pfe: 11.6, ee: 6.8 },
      { day: 'Day 0', pfe: 11.2, ee: 6.5 },
      { day: 'Day +1', pfe: 11.9, ee: 6.9 },
    ],
    sentiment: {
      mean: 0.48,
      change: -0.03,
      summary: 'Slightly softer tone after delays in Vietnamese export permits offset strong turbine demand.',
      drivers: ['Vietnam export permit delay', 'Regional turbine orders', 'Shipping bottleneck easing'],
    },
    liquidity: 'Adequate',
    analystNote: 'Working capital line 63% utilised; watch near-term cash tied to Vietnam receivables.',
    counterparties: ['OCBC Trade Finance', 'Sembcorp Marine', 'VietGrid Partners'],
  },
  {
    id: 'SG-GTC-009',
    name: 'UrbanWave Desalination',
    segment: 'Green Desalination',
    location: 'Singapore · Changi',
    rating: 'Moderate',
    portfolioShare: 0.14,
    limitUtilisation: 0.74,
    exposures: [
      { day: 'Day -5', pfe: 16.2, ee: 9.7 },
      { day: 'Day -4', pfe: 15.8, ee: 9.3 },
      { day: 'Day -3', pfe: 15.1, ee: 8.9 },
      { day: 'Day -2', pfe: 14.7, ee: 8.5 },
      { day: 'Day -1', pfe: 14.2, ee: 8.3 },
      { day: 'Day 0', pfe: 13.9, ee: 8.1 },
      { day: 'Day +1', pfe: 14.6, ee: 8.5 },
    ],
    sentiment: {
      mean: 0.12,
      change: 0.05,
      summary: 'Straits Times coverage praised SG Green Plan compliance; investor chatter remains neutral.',
      drivers: ['Straits Times feature', 'PUB pilot tender', 'Energy intensity targets'],
    },
    liquidity: 'Adequate',
    analystNote: 'Capex ramp for membrane retrofit holds utilisation high; monitor covenant headroom.',
    counterparties: ['UOB Sustainability Desk', 'PUB Water Contracts', 'Temasek GreenTech Fund'],
  },
  {
    id: 'SG-GTC-013',
    name: 'Nimbus Storage Systems',
    segment: 'Grid-scale Batteries',
    location: 'Singapore · Yishun',
    rating: 'Watch',
    portfolioShare: 0.09,
    limitUtilisation: 0.81,
    exposures: [
      { day: 'Day -5', pfe: 18.5, ee: 11.2 },
      { day: 'Day -4', pfe: 18.9, ee: 11.5 },
      { day: 'Day -3', pfe: 19.3, ee: 11.8 },
      { day: 'Day -2', pfe: 19.8, ee: 12.2 },
      { day: 'Day -1', pfe: 20.4, ee: 12.9 },
      { day: 'Day 0', pfe: 21.1, ee: 13.4 },
      { day: 'Day +1', pfe: 22.5, ee: 14.1 },
    ],
    sentiment: {
      mean: -0.18,
      change: -0.06,
      summary: 'Market flagged supplier downgrade; chatter on lithium price swings dampens outlook.',
      drivers: ['Supplier downgrade', 'Lithium volatility', 'Delay in JTC site approval'],
    },
    liquidity: 'Tight',
    analystNote: 'Breaching soft limit on PFE; recommend hedging raw material exposure and revisiting tenor.',
    counterparties: ['Maybank Structured Finance', 'JTC Energy Hub', 'Lithia Materials'],
  },
  {
    id: 'SG-GTC-018',
    name: 'Verde Mobility Services',
    segment: 'EV Fleet & Charging',
    location: 'Singapore · Tampines',
    rating: 'Stressed',
    portfolioShare: 0.07,
    limitUtilisation: 0.95,
    exposures: [
      { day: 'Day -5', pfe: 22.8, ee: 13.5 },
      { day: 'Day -4', pfe: 23.6, ee: 13.9 },
      { day: 'Day -3', pfe: 24.1, ee: 14.6 },
      { day: 'Day -2', pfe: 24.9, ee: 15.1 },
      { day: 'Day -1', pfe: 25.8, ee: 15.8 },
      { day: 'Day 0', pfe: 26.5, ee: 16.4 },
      { day: 'Day +1', pfe: 27.9, ee: 17.2 },
    ],
    sentiment: {
      mean: -0.42,
      change: -0.11,
      summary: 'Sustainability blogs highlight driver churn and EV downtime; credit desks pricing downgrades.',
      drivers: ['Driver attrition reports', 'Battery downtime alerts', 'Rumoured Series B delay'],
    },
    liquidity: 'Tight',
    analystNote: 'Urgent: utilisation near hard limit. Suggest reducing exposure or demanding additional guarantees.',
    counterparties: ['HSBC Sustainable Finance', 'Grab Fleet JV', 'Charge+ Network'],
  },
];

const DashboardList: React.FC = () => {
  const [q, setQ] = useState("");
  const [rating, setRating] = useState<CreditRating | 'ALL'>('ALL');
  const nav = useNavigate();
  const filtered = COMPANIES.filter(c => (rating === 'ALL' || c.rating === rating) && `${c.id} ${c.name} ${c.segment}`.toLowerCase().includes(q.toLowerCase()));

  const formatPct = (value: number) => `${Math.round(value * 100)}%`;

  return (
    <div style={{ background: COLORS.bg }}>
      <Container className="py-10">
        <div className="flex items-end justify-between gap-3 flex-wrap">
          <div>
            <h1 className="text-3xl font-semibold" style={{ color: COLORS.ink }}>Singapore Green Tech Credit Desk</h1>
            <p className="text-sm mt-1" style={{ color: COLORS.sub }}>Select a company to open simulated exposure, utilisation and sentiment analytics.</p>
          </div>
          <div className="flex gap-2 items-center">
            <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search green tech company" className="rounded-full px-4 py-2 text-sm border w-56" style={{ background: COLORS.card, borderColor: COLORS.border }} />
            <select value={rating} onChange={e=>setRating(e.target.value as any)} className="rounded-full px-4 py-2 text-sm border" style={{ background: COLORS.card, borderColor: COLORS.border }}>
              <option value="ALL">All ratings</option>
              {(['Prime','Strong','Moderate','Watch','Stressed'] as CreditRating[]).map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
        </div>
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filtered.map(c => (
            <Card key={c.id} className="hover:shadow-md transition cursor-pointer" title={c.name} right={<RatingBadge rating={c.rating} />}>
              <div className="text-sm space-y-1" style={{ color: COLORS.sub }}>
                <div className="flex items-center justify-between"><span>#{c.id}</span><span>{c.location}</span></div>
                <div>Segment: <b style={{ color: COLORS.ink }}>{c.segment}</b></div>
                <div>Portfolio share: <b style={{ color: COLORS.ink }}>{formatPct(c.portfolioShare)}</b></div>
                <div>Mean sentiment: <b style={{ color: c.sentiment.mean >= 0 ? COLORS.ink : '#7A1C1C' }}>{(c.sentiment.mean >= 0 ? '+' : '') + (c.sentiment.mean * 100).toFixed(0)}%</b></div>
              </div>
              <div className="mt-4 flex items-center justify-between">
                <Button variant="ghost" onClick={()=>nav(`/dashboard/${c.id}` as any)}>Open overview</Button>
                <Button onClick={()=>nav(`/dashboard/${c.id}` as any)}>Simulate PFE</Button>
              </div>
            </Card>
          ))}
        </div>
      </Container>
    </div>
  );
};

const CompanyRisk: React.FC = () => {
  const nav = useNavigate();
  const { id } = useParams();
  const company = COMPANIES.find(c => c.id === id);

  if (!company) {
    return (
      <div style={{ background: COLORS.bg }}>
        <Container className="py-10">
          <Card title="Company not found">
            <p className="text-sm" style={{ color: COLORS.sub }}>We could not locate that green tech counterparty. Please return to the dashboard and pick another entity.</p>
            <div className="mt-4"><Button onClick={()=>nav('/dashboard' as any)}>Back to list</Button></div>
          </Card>
        </Container>
      </div>
    );
  }

  const latest = company.exposures[company.exposures.length - 1];
  const prior = company.exposures[company.exposures.length - 2];
  const pfeDelta = latest.pfe - prior.pfe;
  const eeDelta = latest.ee - prior.ee;
  const utilisationPct = `${Math.round(company.limitUtilisation * 100)}%`;

  const exposureRows = company.exposures.map(point => (
    <tr key={point.day}>
      <td className="py-2 pr-3 text-sm" style={{ color: COLORS.sub }}>{point.day}</td>
      <td className="py-2 pr-3 text-sm font-semibold" style={{ color: COLORS.ink }}>{point.pfe.toFixed(1)}m</td>
      <td className="py-2 text-sm font-semibold" style={{ color: COLORS.ink }}>{point.ee.toFixed(1)}m</td>
    </tr>
  ));

  return (
    <div style={{ background: COLORS.bg }}>
      <Container className="py-10 space-y-6">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-semibold" style={{ color: COLORS.ink }}>{company.name}</h1>
              <RatingBadge rating={company.rating} />
            </div>
            <p className="text-sm mt-1" style={{ color: COLORS.sub }}>{company.segment} · {company.location}</p>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={()=>nav(-1 as any)}>Back</Button>
            <Button>Export snapshot</Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2" title="Simulated Exposure (PFE & EE)">
            <div className="flex items-center gap-6 flex-wrap">
              <div className="rounded-2xl border px-5 py-4 min-w-[220px]" style={{ borderColor: COLORS.border, background: COLORS.card }}>
                <div className="text-xs uppercase tracking-wide" style={{ color: COLORS.sub }}>Latest PFE</div>
                <div className="text-2xl font-semibold mt-1" style={{ color: COLORS.ink }}>{latest.pfe.toFixed(1)}m</div>
                <div className="text-xs mt-1" style={{ color: pfeDelta >= 0 ? '#7A1C1C' : COLORS.ink }}>
                  {pfeDelta >= 0 ? '+' : ''}{pfeDelta.toFixed(1)}m vs prior day
                </div>
                <div className="text-xs mt-3" style={{ color: COLORS.sub }}>Limit utilised: <b style={{ color: COLORS.ink }}>{utilisationPct}</b></div>
              </div>
              <div className="rounded-2xl border px-5 py-4 min-w-[220px]" style={{ borderColor: COLORS.border, background: COLORS.card }}>
                <div className="text-xs uppercase tracking-wide" style={{ color: COLORS.sub }}>Latest EE</div>
                <div className="text-2xl font-semibold mt-1" style={{ color: COLORS.ink }}>{latest.ee.toFixed(1)}m</div>
                <div className="text-xs mt-1" style={{ color: eeDelta >= 0 ? '#7A1C1C' : COLORS.ink }}>
                  {eeDelta >= 0 ? '+' : ''}{eeDelta.toFixed(1)}m vs prior day
                </div>
                <div className="text-xs mt-3" style={{ color: COLORS.sub }}>Liquidity: <b style={{ color: COLORS.ink }}>{company.liquidity}</b></div>
              </div>
            </div>
            <div className="mt-6 overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-xs uppercase tracking-wide" style={{ color: COLORS.sub }}>
                    <th className="pb-2 pr-3 font-medium">Day</th>
                    <th className="pb-2 pr-3 font-medium">PFE (S$ m)</th>
                    <th className="pb-2 font-medium">EE (S$ m)</th>
                  </tr>
                </thead>
                <tbody>
                  {exposureRows}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="Sentiment Pulse">
            <div className="text-sm space-y-3" style={{ color: COLORS.sub }}>
              <div>
                <div>Mean sentiment (7-day): <b style={{ color: company.sentiment.mean >= 0 ? COLORS.ink : '#7A1C1C' }}>{(company.sentiment.mean >= 0 ? '+' : '') + (company.sentiment.mean * 100).toFixed(0)}%</b></div>
                <div>Δ vs prior week: <b style={{ color: company.sentiment.change >= 0 ? COLORS.ink : '#7A1C1C' }}>{(company.sentiment.change >= 0 ? '+' : '') + (company.sentiment.change * 100).toFixed(0)} pts</b></div>
              </div>
              <p>{company.sentiment.summary}</p>
              <div>
                <div className="text-xs uppercase tracking-wide mb-1" style={{ color: COLORS.sub }}>Top signals</div>
                <ul className="list-disc list-inside space-y-1">
                  {company.sentiment.drivers.map(signal => (
                    <li key={signal}>{signal}</li>
                  ))}
                </ul>
              </div>
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card title="Credit Desk Notes">
            <div className="text-sm space-y-3" style={{ color: COLORS.sub }}>
              <p>{company.analystNote}</p>
              <div>
                <div className="text-xs uppercase tracking-wide mb-1" style={{ color: COLORS.sub }}>Counterparties / exposures</div>
                <div className="flex flex-wrap gap-2">
                  {company.counterparties.map(cp => (
                    <span key={cp} className="rounded-full px-3 py-1 text-xs border" style={{ borderColor: COLORS.border, background: COLORS.card }}>{cp}</span>
                  ))}
                </div>
              </div>
            </div>
          </Card>
          <Card title="Actions">
            <div className="text-sm space-y-3" style={{ color: COLORS.sub }}>
              <p>Trigger follow-ups for utilisation ≥ 80% and negative sentiment trends. Use simulated exposure data to pre-fill VaR stress scenarios for Singapore credit committees.</p>
              <div className="flex gap-2 flex-wrap">
                <Button variant="ghost">Run stress scenario</Button>
                <Button variant="ghost">Push to CRM</Button>
              </div>
              <p className="text-xs">Sentiment powered by CQ2 NLP on sustainability market news feeds.</p>
            </div>
          </Card>
        </div>
      </Container>
    </div>
  );
};

// ===== Smoke Tests =====
function runSmokeTests() {
  // 1) COLORS keys present
  const must = ['bg','card','ink','sub','border','mint','mintDeep'];
  const miss = must.filter(k => !(k in COLORS));
  if (miss.length) console.warn('Missing color tokens:', miss);
  // 2) Asset paths are strings
  Object.entries(ASSETS).forEach(([k,v]) => { if (typeof v !== 'string') console.warn('Asset not a string', k); });
  // 3) Components exist
  const comps = { Container, Button, Card, SiteHeader } as any;
  Object.entries(comps).forEach(([n,c]) => { if (typeof c !== 'function') console.warn('Component is not a function:', n); });
}
runSmokeTests();

// ===== App wrapper =====
export default function CQ2AtlasSite_Enriched() {
  return (
    <BrowserRouter>
      <div className="min-h-screen" style={{ background: COLORS.bg, color: COLORS.ink }}>
        <SiteHeader />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/team" element={<TeamPage />} />
          <Route path="/dashboard" element={<DashboardList />} />
          <Route path="/dashboard/:id" element={<CompanyRisk />} />
        </Routes>
        <SiteFooter />
      </div>
    </BrowserRouter>
  );
}

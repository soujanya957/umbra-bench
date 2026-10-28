"use client";

import { useEffect, useRef, useState, type PointerEvent } from "react";

/**
 * UMBRA hero.
 *
 * A dark stage under a real pendant spotlight. Pull the cord to turn on the
 * light: the bloom blooms on, dust rises through it, "UMBR" resolves, and a
 * robot arm swings in from the top-right carrying the letter "A", setting it
 * down to complete "UMBRA" — the word and the A casting a clear soft shadow.
 *
 * Rendering: the word + shadow and the arm + A live in two overlaid SVGs that
 * share the SAME viewBox + preserveAspectRatio, so the A always lands exactly
 * after UMBR. Self-contained (canvas particles + styled-jsx), no external deps.
 */

/* --- 2-link arm kinematics (viewBox coordinates) ------------------------- *
 * Base joint at (930,150); at rest the elbow sits at (800,230) and the gripper
 * (holding the A) at (678,304). m1 rotates the whole arm about the base, m2
 * rotates the forearm about the elbow. We drive them by inverse kinematics so
 * the gripper — and the letter + its shadow it carries — follows the cursor.  */
const ARM_BASE = { x: 930, y: 150 };
const ARM_U0 = { x: -130, y: 80 }; // base -> elbow at rest
const ARM_F0 = { x: -122, y: 74 }; // elbow -> gripper at rest
const ARM_L1 = Math.hypot(ARM_U0.x, ARM_U0.y);
const ARM_L2 = Math.hypot(ARM_F0.x, ARM_F0.y);
const ARM_PHI1 = Math.atan2(ARM_U0.y, ARM_U0.x);
const ARM_PHI2 = Math.atan2(ARM_F0.y, ARM_F0.x);
const RAD2DEG = 180 / Math.PI;

// gripper position for given joint angles (degrees) — forward kinematics
function gripperAt(a1deg: number, a2deg: number) {
  const alpha = ARM_PHI1 + a1deg / RAD2DEG;
  const ex = ARM_BASE.x + ARM_L1 * Math.cos(alpha);
  const ey = ARM_BASE.y + ARM_L1 * Math.sin(alpha);
  const beta = ARM_PHI2 + (a1deg + a2deg) / RAD2DEG;
  return { x: ex + ARM_L2 * Math.cos(beta), y: ey + ARM_L2 * Math.sin(beta) };
}

// joint angles (degrees) that put the gripper at/toward (tx,ty) — inverse kin.
function solveArm(tx: number, ty: number) {
  const dx = tx - ARM_BASE.x;
  const dy = ty - ARM_BASE.y;
  let D = Math.hypot(dx, dy) || 0.001;
  D = Math.max(Math.abs(ARM_L1 - ARM_L2) + 1, Math.min(ARM_L1 + ARM_L2, D));
  const g = Math.atan2(dy, dx);
  const cx = ARM_BASE.x + D * Math.cos(g); // clamped target (if out of reach)
  const cy = ARM_BASE.y + D * Math.sin(g);
  const cosA = Math.max(-1, Math.min(1, (ARM_L1 * ARM_L1 + D * D - ARM_L2 * ARM_L2) / (2 * ARM_L1 * D)));
  const alpha = g + Math.acos(cosA); // elbow-up branch (natural for this mount)
  const ex = ARM_BASE.x + ARM_L1 * Math.cos(alpha);
  const ey = ARM_BASE.y + ARM_L1 * Math.sin(alpha);
  const beta = Math.atan2(cy - ey, cx - ex);
  const a1 = (alpha - ARM_PHI1) * RAD2DEG;
  const a2 = (beta - ARM_PHI2) * RAD2DEG - a1;
  return { a1, a2 };
}

interface Particle {
  x: number;
  y: number;
  speed: number;
  opacity: number;
  fadeStart: number;
  fadingOut: boolean;
  reset: () => void;
  update: () => void;
  draw: (ctx: CanvasRenderingContext2D, lit: boolean) => void;
}

export default function LightCordHero() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const rafRef = useRef<number>();
  const litRef = useRef(false);

  const [lightOn, setLightOn] = useState(false);
  const [pulling, setPulling] = useState(false);
  const [swinging, setSwinging] = useState(false);
  // false = arm still sweeping in to grab the A; true = grabbed → pick-up loop
  const [engaged, setEngaged] = useState(false);

  useEffect(() => {
    litRef.current = lightOn;
  }, [lightOn]);

  // --- arm control ---------------------------------------------------------
  // manual === true means the inline joint angles drive the arm (dragging, or
  // parked/frozen); manual === false means the CSS pick-up loop plays.
  const armSvgRef = useRef<SVGSVGElement>(null);
  const m1Ref = useRef<SVGGElement>(null);
  const m2Ref = useRef<SVGGElement>(null);
  const [manual, setManual] = useState(false);
  const [armAngles, setArmAngles] = useState({ a1: 0, a2: 0 });
  const armDragRef = useRef(false);
  const armAnglesRef = useRef({ a1: 0, a2: 0 });
  const grabOffsetRef = useRef({ x: 0, y: 0 });
  const idleTimerRef = useRef<number>();
  const resumeTimerRef = useRef<number>();
  const tweenRafRef = useRef<number>();

  useEffect(() => {
    armAnglesRef.current = armAngles;
  }, [armAngles]);

  const setAngles = (a: { a1: number; a2: number }) => {
    armAnglesRef.current = a;
    setArmAngles(a);
  };

  // read the arm's current rotation (deg) straight from the live CSS transforms,
  // so handing off from the animation to manual control never jumps
  const readLiveAngles = () => {
    const parse = (el: SVGGElement | null) => {
      if (!el) return 0;
      const t = getComputedStyle(el).transform;
      const m = t && t.startsWith("matrix") ? t.slice(t.indexOf("(") + 1).split(",").map(Number) : null;
      return m ? Math.atan2(m[1], m[0]) * RAD2DEG : 0;
    };
    return { a1: parse(m1Ref.current), a2: parse(m2Ref.current) };
  };

  // ease the arm back to its resting pose, then run `done`
  const glideToRest = (done?: () => void) => {
    cancelAnimationFrame(tweenRafRef.current!);
    const start = { ...armAnglesRef.current };
    const t0 = performance.now();
    const dur = 700;
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3); // easeOutCubic
      setAngles({ a1: start.a1 * (1 - e), a2: start.a2 * (1 - e) });
      if (k < 1) tweenRafRef.current = requestAnimationFrame(step);
      else if (done) done();
    };
    tweenRafRef.current = requestAnimationFrame(step);
  };

  // after 10s with no interaction, glide to rest and freeze there (grabbable)
  const scheduleIdleFreeze = () => {
    window.clearTimeout(idleTimerRef.current);
    idleTimerRef.current = window.setTimeout(() => {
      setAngles(readLiveAngles());
      setManual(true);
      glideToRest();
    }, 10000);
  };

  // on light-on: sweep in, grab the A, start the pick-up loop, then idle-freeze
  useEffect(() => {
    if (!lightOn) {
      window.clearTimeout(idleTimerRef.current);
      window.clearTimeout(resumeTimerRef.current);
      cancelAnimationFrame(tweenRafRef.current!);
      setEngaged(false);
      setManual(false);
      setAngles({ a1: 0, a2: 0 });
      return;
    }
    const t = window.setTimeout(() => {
      setEngaged(true);
      scheduleIdleFreeze();
    }, 2600);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lightOn]);

  // client pixel -> SVG viewBox coordinate
  const toViewBox = (clientX: number, clientY: number) => {
    const svg = armSvgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    return pt.matrixTransform(ctm.inverse());
  };

  const onArmPointerDown = (e: PointerEvent<SVGCircleElement>) => {
    if (!lightOn || !engaged) return; // only grabbable once it holds the A
    const p = toViewBox(e.clientX, e.clientY);
    if (!p) return;
    e.preventDefault();
    window.clearTimeout(idleTimerRef.current);
    window.clearTimeout(resumeTimerRef.current);
    cancelAnimationFrame(tweenRafRef.current!);
    const live = readLiveAngles(); // take over from wherever it is, no jump
    setAngles(live);
    setManual(true);
    armDragRef.current = true;
    const grip = gripperAt(live.a1, live.a2);
    grabOffsetRef.current = { x: grip.x - p.x, y: grip.y - p.y };
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {}
  };
  const onArmPointerMove = (e: PointerEvent<SVGCircleElement>) => {
    if (!armDragRef.current) return;
    const p = toViewBox(e.clientX, e.clientY);
    if (!p) return;
    setAngles(solveArm(p.x + grabOffsetRef.current.x, p.y + grabOffsetRef.current.y));
  };
  const onArmPointerEnd = () => {
    if (!armDragRef.current) return;
    armDragRef.current = false;
    // stay parked where dropped for at least 5s, then glide back and replay
    window.clearTimeout(resumeTimerRef.current);
    resumeTimerRef.current = window.setTimeout(() => {
      glideToRest(() => {
        setManual(false); // hand back to the CSS loop, which starts at rest
        scheduleIdleFreeze();
      });
    }, 5000);
  };

  const makeParticle = (canvas: HTMLCanvasElement): Particle => {
    const p: Particle = {
      x: 0,
      y: 0,
      speed: 0,
      opacity: 1,
      fadeStart: 0,
      fadingOut: false,
      reset() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.speed = Math.random() / 5 + 0.1;
        this.opacity = 1;
        this.fadeStart = Date.now() + Math.random() * 600 + 100;
        this.fadingOut = false;
      },
      update() {
        this.y -= this.speed;
        if (this.y < 0) this.reset();
        if (!this.fadingOut && Date.now() > this.fadeStart) this.fadingOut = true;
        if (this.fadingOut) {
          this.opacity -= 0.008;
          if (this.opacity <= 0) this.reset();
        }
      },
      draw(ctx, lit) {
        const glow = lit ? this.opacity : this.opacity * 0.14;
        ctx.fillStyle = `rgba(${255 - (Math.random() * 255) / 2}, 255, 255, ${glow})`;
        ctx.fillRect(this.x, this.y, 0.5, Math.random() * 2 + 1);
      },
    };
    p.reset();
    p.y = Math.random() * canvas.height;
    return p;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const init = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      const count = Math.floor((canvas.width * canvas.height) / 6500);
      particlesRef.current = Array.from({ length: count }, () => makeParticle(canvas));
    };
    const loop = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const p of particlesRef.current) {
        p.update();
        p.draw(ctx, litRef.current);
      }
      rafRef.current = requestAnimationFrame(loop);
    };

    init();
    loop();
    window.addEventListener("resize", init);
    return () => {
      window.removeEventListener("resize", init);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [dragY, setDragY] = useState(0);
  const [dragging, setDragging] = useState(false);
  const draggingRef = useRef(false);
  const startYRef = useRef(0);
  const maxDragRef = useRef(0);

  // a plain click (or keyboard): play the full pull-down + slow settle, then toggle
  const clickPull = () => {
    if (pulling) return;
    setPulling(true);
    setSwinging(true);
    window.setTimeout(() => setLightOn((v) => !v), 260);
    window.setTimeout(() => setPulling(false), 900);
    window.setTimeout(() => setSwinging(false), 1200);
  };

  // drag: grab the knob, pull the cord down, release to fire
  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (pulling) return;
    draggingRef.current = true;
    startYRef.current = e.clientY;
    maxDragRef.current = 0;
    setDragging(true);
    setDragY(0);
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {}
  };
  const onPointerMove = (e: PointerEvent<HTMLButtonElement>) => {
    if (!draggingRef.current) return;
    const dy = Math.max(0, Math.min(90, e.clientY - startYRef.current));
    maxDragRef.current = Math.max(maxDragRef.current, dy);
    setDragY(dy);
  };
  const onPointerEnd = () => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    const maxd = maxDragRef.current;
    setDragging(false); // re-enables the slow spring-back transition
    setDragY(0);
    if (maxd < 5) {
      clickPull(); // barely moved → treat as a click
    } else if (maxd >= 24) {
      window.setTimeout(() => setLightOn((v) => !v), 220); // pulled far enough
    }
  };

  return (
    <section className={`stage ${lightOn ? "is-lit" : ""} ${engaged ? "is-engaged" : ""} ${manual ? "is-manual" : ""}`}>
      <canvas ref={canvasRef} className="particles" />

      <div className="bloom" />
      <div className="vignette" />

      {/* pendant spotlight fixture (always visible) */}
      <svg className="scene lamp-svg" viewBox="0 0 1000 640" preserveAspectRatio="xMidYMid meet" fill="none" aria-hidden="true">
        <defs>
          <linearGradient id="shade" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#0d0d11" />
            <stop offset="30%" stopColor="#2a2b31" />
            <stop offset="55%" stopColor="#3a3b42" />
            <stop offset="100%" stopColor="#0a0a0d" />
          </linearGradient>
          {/* warm glow spilling out of the shade opening */}
          <radialGradient id="bulbHalo" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ffeabf" stopOpacity="0.9" />
            <stop offset="55%" stopColor="#ffdf9e" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#ffdf9e" stopOpacity="0" />
          </radialGradient>
          {/* the glowing opening / bulb, hot in the middle */}
          <radialGradient id="bulbGlow" cx="50%" cy="42%" r="60%">
            <stop offset="0%" stopColor="#fffdf7" />
            <stop offset="38%" stopColor="#fff0cc" />
            <stop offset="100%" stopColor="#e6c98d" />
          </radialGradient>
          <filter id="soft" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="7" />
          </filter>
        </defs>
        <g transform="translate(486,30) scale(0.82) translate(-486,-30)">
          <ellipse cx="486" cy="30" rx="126" ry="13" fill="#0c0c0f" />
          <rect x="478" y="30" width="16" height="26" fill="#101015" />
          <path d="M446 56 L526 56 L540 176 Q486 196 432 176 Z" fill="url(#shade)" />
          <path d="M452 60 L462 60 L452 174 Q447 172 444 168 Z" fill="#54555d" opacity="0.5" />
          {/* soft halo, then the glowing opening, then a hot filament core */}
          <ellipse className="aperture" cx="486" cy="182" rx="98" ry="42" fill="url(#bulbHalo)" filter="url(#soft)" />
          <ellipse className="aperture" cx="486" cy="179" rx="54" ry="17" fill="url(#bulbGlow)" />
          <ellipse className="aperture" cx="486" cy="177" rx="15" ry="6" fill="#fffef9" filter="url(#soft)" />
        </g>
      </svg>

      {/* the word + its cast shadow (revealed with the light) */}
      <svg className="scene word-svg" viewBox="0 0 1000 640" preserveAspectRatio="xMidYMid meet" fill="none" aria-hidden="true">
        <defs>
          {/* metallic, top-lit blue-grey used for every letter (defined once, shared) */}
          <linearGradient id="letterGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f0f4fb" />
            <stop offset="24%" stopColor="#c2cde0" />
            <stop offset="52%" stopColor="#93a2bc" />
            <stop offset="78%" stopColor="#616f8a" />
            <stop offset="100%" stopColor="#354052" />
          </linearGradient>
        </defs>
        <g transform="translate(0,126) skewX(-9) scale(1,0.72)" style={{ filter: "blur(4px)" }}>
          <text x="220" y="392" style={{ fontFamily: "var(--font-cinzel), Georgia, serif" }}fontSize="120" letterSpacing="10" fill="rgba(6,7,10,0.55)">
            UMBR
          </text>
        </g>
        <text x="220" y="384" style={{ fontFamily: "var(--font-cinzel), Georgia, serif" }}fontSize="120" letterSpacing="10" fill="url(#letterGrad)">
          UMBR
        </text>

        {/* the A resting in place — shown until the arm grabs it, then it hands
            off to the identical A carried by the arm (see .arm-a) */}
        <g className="rest-a">
          <g transform="translate(0,126) skewX(-9) scale(1,0.72)" style={{ filter: "blur(4px)" }}>
            <text x="640" y="392" style={{ fontFamily: "var(--font-cinzel), Georgia, serif" }} fontSize="120" fill="rgba(6,7,10,0.55)">
              A
            </text>
          </g>
          <text x="640" y="384" style={{ fontFamily: "var(--font-cinzel), Georgia, serif" }} fontSize="120" fill="url(#letterGrad)">
            A
          </text>
        </g>
      </svg>

      {/* robot arm carrying the A — base joint fixed; elbow (m1) + end effector (m2) swing */}
      <svg
        ref={armSvgRef}
        className="scene arm-svg"
        viewBox="0 0 1000 640"
        preserveAspectRatio="xMidYMid meet"
        fill="none"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="gm" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#61636b" />
            <stop offset="34%" stopColor="#2c2d33" />
            <stop offset="62%" stopColor="#3c3d44" />
            <stop offset="100%" stopColor="#131418" />
          </linearGradient>
          <linearGradient id="cw" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#cfd2d8" />
            <stop offset="50%" stopColor="#83868d" />
            <stop offset="100%" stopColor="#494b51" />
          </linearGradient>
          <radialGradient id="jn" cx="36%" cy="32%" r="70%">
            <stop offset="0%" stopColor="#7d7f87" />
            <stop offset="55%" stopColor="#303138" />
            <stop offset="100%" stopColor="#0e0e12" />
          </radialGradient>
        </defs>

        {/* fixed base: mount to the top-right edge + base joint (does not move) */}
        <path d="M1010 96 L930 150 L946 168 L1020 120 Z" fill="#17181d" />
        <circle cx="930" cy="150" r="20" fill="url(#jn)" stroke="#000" strokeWidth="1" />
        <circle cx="930" cy="150" r="6" fill="#06060a" />

        {/* everything below the base rotates around the base joint */}
        <g ref={m1Ref} className="m1" style={manual ? { transform: `rotate(${armAngles.a1}deg)` } : undefined}>
          <path d="M918 162 q-30 22 -8 46 q20 22 -12 44" stroke="#141418" strokeWidth="5" fill="none" opacity="0.7" />
          <path d="M930 150 L800 230" stroke="url(#gm)" strokeWidth="24" strokeLinecap="round" />
          <path d="M926 154 L800 232" stroke="#9092a0" strokeWidth="2.5" strokeLinecap="round" opacity="0.5" />
          <circle cx="800" cy="230" r="18" fill="url(#jn)" stroke="#000" strokeWidth="1" />
          <circle cx="800" cy="230" r="5" fill="#06060a" />

          {/* forearm + end effector rotate around the elbow */}
          <g ref={m2Ref} className="m2" style={manual ? { transform: `rotate(${armAngles.a2}deg)` } : undefined}>
            <path d="M800 230 L678 304" stroke="url(#gm)" strokeWidth="19" strokeLinecap="round" />
            <path d="M800 232 L678 306" stroke="#828492" strokeWidth="2" strokeLinecap="round" opacity="0.5" />

            {/* the A the arm carries — hidden during the approach, revealed the
                instant it grabs the resting A (they sit at the same spot) */}
            <g className="arm-a">
              {/* A's cast shadow (kept dark for emphasis) */}
              <g transform="translate(0,126) skewX(-9) scale(1,0.72)" style={{ filter: "blur(4px)" }}>
                <text x="640" y="392" style={{ fontFamily: "var(--font-cinzel), Georgia, serif" }}fontSize="120" fill="rgba(6,7,10,0.55)">
                  A
                </text>
              </g>
              <text x="640" y="384" style={{ fontFamily: "var(--font-cinzel), Georgia, serif" }}fontSize="120" fill="url(#letterGrad)">
                A
              </text>
            </g>

            {/* metallic claw gripping the upper part of the A */}
            <circle cx="678" cy="304" r="12" fill="url(#cw)" stroke="#33353b" strokeWidth="1" />
            <g stroke="url(#cw)" strokeWidth="11" fill="none" strokeLinecap="round">
              <path d="M678 304 q-27 5 -32 32" />
              <path d="M678 304 q25 12 21 40" />
            </g>
            <circle cx="648" cy="338" r="5.5" fill="#c2c5cb" />
            <circle cx="700" cy="346" r="5.5" fill="#c2c5cb" />

            {/* invisible, generously sized drag handle over the gripper */}
            <circle
              className="grip-handle"
              cx="678"
              cy="322"
              r="48"
              fill="none"
              onPointerDown={onArmPointerDown}
              onPointerMove={onArmPointerMove}
              onPointerUp={onArmPointerEnd}
              onPointerCancel={onArmPointerEnd}
            />
          </g>
        </g>
      </svg>

      {/* hanging pull cord — click, or grab the knob and drag it down */}
      <button
        aria-label={lightOn ? "Pull cord to turn the light off" : "Pull cord to turn the light on"}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            clickPull();
          }
        }}
        className={`cord ${swinging ? "cord-swinging" : ""} ${dragging ? "is-dragging" : ""}`}
      >
        <span
          className="cordline"
          style={{
            height: `calc(5.5em + ${dragY}px)`,
            transition: dragging ? "none" : "height 0.9s cubic-bezier(0.33, 1.25, 0.5, 1)",
          }}
        />
        <span className="knob" />
      </button>

      {!lightOn && (
        <div className="hint">
          <p className="hint-1">pull the cord</p>
          <p className="hint-2">turn on the light</p>
        </div>
      )}

      {/* subtitle + scroll cue */}
      <div className="undertitle">
        <p className="subtitle">Choreographing multi-robot arms to sculpt light into designed shadows.</p>
        <div className="navlinks">
          <a href="#abstract">Project overview</a>
          <a href="#video">Demos</a>
          <a href="#design">The design tool</a>
          <a href={`${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/benchmark/index.html`}>Benchmark</a>
        </div>
        <a href="#abstract" className="scroll-hint" aria-label="Scroll to content">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </a>
      </div>

      <style jsx>{`
        .stage {
          position: relative;
          height: 100vh;
          width: 100%;
          overflow: hidden;
          background: #050505;
        }
        .particles {
          position: absolute;
          inset: 0;
          height: 100%;
          width: 100%;
          pointer-events: none;
          z-index: 1;
        }
        .scene {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
        }

        .bloom {
          position: absolute;
          left: 50%;
          top: 52%;
          transform: translate(-50%, -50%);
          width: 108%;
          height: 128%;
          border-radius: 50%;
          background: radial-gradient(
            ellipse at 50% 46%,
            rgba(252, 250, 244, 0.97) 0%,
            rgba(247, 240, 226, 0.62) 26%,
            rgba(236, 223, 196, 0.3) 46%,
            rgba(176, 164, 138, 0.1) 66%,
            transparent 80%
          );
          filter: blur(10px);
          opacity: 0;
          transition: opacity 1.6s ease;
          z-index: 2;
        }
        .is-lit .bloom {
          opacity: 1;
        }
        .vignette {
          position: absolute;
          inset: 0;
          pointer-events: none;
          z-index: 3;
          background: radial-gradient(
            ellipse at 50% 54%,
            transparent 30%,
            rgba(3, 3, 4, 0.5) 60%,
            rgba(2, 2, 3, 0.96) 90%
          );
          opacity: 0;
          transition: opacity 1.6s ease;
        }
        .is-lit .vignette {
          opacity: 1;
        }

        .lamp-svg {
          z-index: 6;
          pointer-events: none; /* decorative — must not steal the arm's clicks */
        }
        .aperture {
          opacity: 0.15;
          transition: opacity 1.4s ease;
        }
        .is-lit .aperture {
          opacity: 1;
        }

        .word-svg {
          z-index: 4;
          opacity: 0;
          transition: opacity 1.2s ease 0.25s;
          pointer-events: none; /* decorative */
        }
        .is-lit .word-svg {
          opacity: 1;
        }

        .arm-svg {
          z-index: 5;
          opacity: 0;
          transition: opacity 0.8s ease;
        }
        .is-lit .arm-svg {
          opacity: 1;
        }
        /* the gripper is the drag handle — active only once it holds the A */
        .grip-handle {
          pointer-events: none;
        }
        .is-engaged .grip-handle {
          pointer-events: all;
          cursor: grab;
        }
        .is-manual .grip-handle {
          cursor: grabbing;
        }
        /* upper arm swings around the fixed base joint; forearm around the elbow */
        .m1 {
          transform-box: view-box;
          transform-origin: 930px 150px;
        }
        .m2 {
          transform-box: view-box;
          transform-origin: 800px 230px;
        }
        /* hand control to the inline IK transforms while the user is dragging */
        .is-manual .m1,
        .is-manual .m2 {
          animation: none !important;
        }
        /* 1) APPROACH: the arm sweeps in from a folded-up pose down to the A and
           settles on it (transform held via forwards until it engages). */
        .is-lit .m1 {
          animation: armIn1 2.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
        .is-lit .m2 {
          animation: armIn2 2.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
        @keyframes armIn1 {
          from { transform: rotate(-30deg); }
          to { transform: rotate(0deg); }
        }
        @keyframes armIn2 {
          from { transform: rotate(54deg); }
          to { transform: rotate(0deg); }
        }
        /* 2) GRABBED: swap to the infinite pick-up loop. Both loops start/end at
           rest (rotate 0) so there's no jump from the held approach pose:
           hold, lift the A away, hold, set it back, repeat. */
        .is-lit.is-engaged .m1 {
          animation: swing1 6s ease-in-out infinite;
        }
        .is-lit.is-engaged .m2 {
          animation: swing2 6s ease-in-out infinite;
        }
        @keyframes swing1 {
          0%,
          100% {
            transform: rotate(0deg);
          }
          20%,
          52% {
            transform: rotate(-14deg);
          }
          72% {
            transform: rotate(0deg);
          }
        }
        @keyframes swing2 {
          0%,
          100% {
            transform: rotate(0deg);
          }
          20%,
          52% {
            transform: rotate(30deg);
          }
          72% {
            transform: rotate(0deg);
          }
        }

        /* the A hands off from its resting spot to the arm at the grab moment.
           Identical position + a quick crossfade makes it read as one letter. */
        .rest-a,
        .arm-a {
          transition: opacity 0.3s ease;
        }
        .rest-a {
          opacity: 0;
        }
        .is-lit .rest-a {
          opacity: 1;
        }
        .is-lit.is-engaged .rest-a {
          opacity: 0;
        }
        .arm-a {
          opacity: 0;
        }
        .is-lit.is-engaged .arm-a {
          opacity: 1;
        }

        /* pull cord */
        .cord {
          position: absolute;
          top: 0;
          right: 14%;
          z-index: 20;
          display: flex;
          flex-direction: column;
          align-items: center;
          cursor: pointer;
          background: none;
          border: none;
          padding: 0;
          cursor: grab;
        }
        .cord.is-dragging {
          cursor: grabbing;
        }
        /* click/keyboard: the wire stretches down (anchored at the top of the
           screen), then slowly draws back up to rest */
        .cord-swinging .cordline {
          animation: cordPull 1.2s cubic-bezier(0.33, 1.2, 0.5, 1);
        }
        @keyframes cordPull {
          0% { height: 5.5em; }
          20% { height: 10em; }
          70% { height: 5.2em; }
          100% { height: 5.5em; }
        }
        @media (prefers-reduced-motion: reduce) {
          .cord-swinging .cordline { animation: none; }
        }
        .cordline {
          display: block;
          width: 2px;
          height: 5.5em;
          background: linear-gradient(180deg, rgba(220, 226, 236, 0.06), rgba(220, 226, 236, 0.5));
        }
        .knob {
          display: block;
          width: 0.85em;
          height: 0.85em;
          border-radius: 50%;
          background: radial-gradient(circle at 35% 30%, #f2f5fb, #8a8d94);
          box-shadow: 0 0 0.6em rgba(230, 232, 238, 0.5);
          transition: box-shadow 0.4s ease;
        }
        .cord:hover .knob {
          box-shadow: 0 0 1em rgba(255, 244, 214, 0.85);
        }

        .hint {
          position: absolute;
          top: 4.4em;
          right: calc(14% + 1.8em);
          z-index: 20;
          text-align: right;
          line-height: 1.25;
          pointer-events: none;
        }
        .hint-1 {
          font-size: 0.85rem;
          font-weight: 500;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          color: #ffffff;
        }
        .hint-2 {
          font-size: 0.7rem;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          color: rgba(255, 255, 255, 0.75);
        }

        .undertitle {
          position: absolute;
          left: 0;
          right: 0;
          top: 72%;
          z-index: 10;
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: 0 1.5rem;
        }
        .subtitle {
          max-width: 34rem;
          font-size: clamp(0.95rem, 1.4vw, 1.15rem);
          letter-spacing: 0.01em;
          color: #eef1f6;
          text-shadow: 0 1px 14px rgba(0, 0, 0, 0.7);
          opacity: 0;
          transform: translateY(0.6em);
          transition: opacity 1s ease 1.6s, transform 1s ease 1.6s;
        }
        .is-lit .subtitle {
          opacity: 1;
          transform: translateY(0);
        }
        .navlinks {
          margin-top: 1.4rem;
          display: flex;
          flex-wrap: wrap;
          justify-content: center;
          gap: 0.6rem;
          opacity: 1;            /* permanent — shown even before the light is on */
          transform: none;
        }
        .is-lit .navlinks {
          opacity: 1;
          transform: translateY(0);
        }
        .navlinks a {
          border-radius: 9999px;
          border: 1px solid rgba(255, 255, 255, 0.2);
          background: rgba(8, 10, 16, 0.4);
          padding: 0.45rem 0.95rem;
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: rgba(235, 240, 248, 0.9);
          backdrop-filter: blur(4px);
          transition: border-color 0.3s ease, background 0.3s ease, color 0.3s ease;
        }
        .navlinks a:hover {
          border-color: rgba(255, 255, 255, 0.45);
          background: rgba(255, 255, 255, 0.12);
          color: #ffffff;
        }
        .scroll-hint {
          margin-top: 1.8rem;
          color: rgba(230, 234, 242, 0.7);
          opacity: 0;
          transition: opacity 1s ease 2s;
          animation: float 2.4s ease-in-out infinite;
        }
        .is-lit .scroll-hint {
          opacity: 1;
        }
        @keyframes float {
          0%,
          100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(0.4em);
          }
        }
      `}</style>
    </section>
  );
}

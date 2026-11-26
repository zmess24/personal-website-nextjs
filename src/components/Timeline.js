import { useEffect, useRef } from "react";
import Link from "next/link";
import StripeMark from "./StripeMark";
import { LANE_UP, LANE_DN } from "@/lib/timeline";

const KIND_CLASS = { post: "post", proj: "proj", now: "proj now", job: "wk" };

function EventLink({ e, className, style, children }) {
	if (e.href.startsWith("/")) {
		return (
			<Link className={className} style={style} href={e.href}>
				{children}
			</Link>
		);
	}
	return (
		<a className={className} style={style} href={e.href} target="_blank" rel="noreferrer">
			{children}
		</a>
	);
}

// One line, oldest to newest. Writing hangs above it, things I built and work sit below,
// and it ends at what I'm building now. On phones it becomes a vertical list (TimelineList).
export function TimelineRail({ layout }) {
	const ref = useRef(null);

	useEffect(() => {
		const r = ref.current;
		if (!r) return;
		// Open at the recent end so the newest work is in the first frame.
		r.scrollLeft = r.scrollWidth;

		// Drag to scrub with a mouse; a drag doesn't count as a click.
		let down = false,
			sx = 0,
			sl = 0,
			moved = false;
		const onDown = (e) => {
			if (e.pointerType !== "mouse") return;
			down = true;
			moved = false;
			sx = e.clientX;
			sl = r.scrollLeft;
		};
		const onMove = (e) => {
			if (!down) return;
			const d = e.clientX - sx;
			if (Math.abs(d) > 4) moved = true;
			r.scrollLeft = sl - d;
		};
		const onUp = () => (down = false);
		const onClick = (e) => {
			if (moved) {
				e.preventDefault();
				moved = false;
			}
		};
		// A vertical wheel scrolls the line sideways until it reaches an end, then the page scrolls.
		const onWheel = (e) => {
			if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
			const max = r.scrollWidth - r.clientWidth;
			if ((e.deltaY < 0 && r.scrollLeft <= 0) || (e.deltaY > 0 && r.scrollLeft >= max - 1)) return;
			r.scrollLeft += e.deltaY;
			e.preventDefault();
		};
		r.addEventListener("pointerdown", onDown);
		window.addEventListener("pointermove", onMove);
		window.addEventListener("pointerup", onUp);
		r.addEventListener("click", onClick, true);
		r.addEventListener("wheel", onWheel, { passive: false });
		return () => {
			r.removeEventListener("pointerdown", onDown);
			window.removeEventListener("pointermove", onMove);
			window.removeEventListener("pointerup", onUp);
			r.removeEventListener("click", onClick, true);
			r.removeEventListener("wheel", onWheel);
		};
	}, []);

	const { events, years, width, height, axis } = layout;
	const vars = { "--ax": `${axis}px`, "--lh": `${LANE_UP}px`, "--ld": `${LANE_DN}px`, width: `${width}px`, height: `${height}px` };

	return (
		<div className="rail" id="line" ref={ref} tabIndex={0} aria-label={`Timeline, ${layout.startYear} to now`}>
			<div className="rail-in" style={vars}>
				<div className="axis" />
				{years.map((y) => (
					<span key={y.year} className="yr" style={{ "--x": `${y.x}px` }}>
						{y.year}
					</span>
				))}
				{events.map((e) => (
					<EventLink
						key={`${e.kind}-${e.title}-${e.date}`}
						e={e}
						className={`ev ${KIND_CLASS[e.kind]} ${e.kind === "post" ? "up" : "dn"}`}
						style={{ "--x": `${e.x}px`, "--lane": e.lane }}
					>
						<span className="dt">
							{e.logo === "stripe" && <StripeMark />}
							{e.label}
						</span>
						<span className="tt">{e.title}</span>
						<span className="su">{e.subtitle}</span>
						{e.status && <span className="st">{e.status}</span>}
					</EventLink>
				))}
			</div>
		</div>
	);
}

export function TimelineList({ layout }) {
	const items = [...layout.events].reverse();
	return (
		<ol className="vlist">
			{items.map((e) => (
				<li key={`${e.kind}-${e.title}-${e.date}`} className={KIND_CLASS[e.kind]}>
					{e.logo === "stripe" && (
						<span className="vm">
							<StripeMark />
						</span>
					)}
					<EventLink e={e}>
						<span className="dt">{e.label}</span>
						<span className="tt">{e.title}</span>
						<span className="su">{e.subtitle}</span>
					</EventLink>
				</li>
			))}
		</ol>
	);
}

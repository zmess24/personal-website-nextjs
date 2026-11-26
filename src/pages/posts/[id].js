import { useEffect, useRef } from "react";
import Head from "next/head";
import Link from "next/link";
import moment from "moment";
import { getAllPostIds, getPostData, getSortedPostsData } from "@/lib/posts";
import { getSortedProjectsData } from "@/lib/projects";
import { postStrip } from "@/lib/timeline";
import { NOW, SITE_URL } from "@/data/site";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import SwishJam from "@/components/SwishJam";

const fmt = (d, f) => moment(d, "MM-DD-YYYY").format(f);

// Copy buttons, reading progress along the top axis, and the contents list following the section you're reading.
function usePostEffects(articleRef, barRef, tocRef) {
	useEffect(() => {
		const art = articleRef.current;
		if (!art) return;

		const onCopy = async (e) => {
			const b = e.target.closest(".cp");
			if (!b) return;
			const code = b.closest("figure").querySelector("code");
			try {
				await navigator.clipboard.writeText(code.textContent);
				b.textContent = "Copied";
				b.classList.add("ok");
			} catch (_) {
				b.textContent = "Select";
			}
			setTimeout(() => {
				b.textContent = "Copy";
				b.classList.remove("ok");
			}, 1600);
		};
		art.addEventListener("click", onCopy);

		const links = tocRef.current ? [...tocRef.current.querySelectorAll("a")] : [];
		const heads = links.map((a) => document.getElementById(a.getAttribute("href").slice(1)));
		let tick = false;
		const update = () => {
			tick = false;
			const r = art.getBoundingClientRect();
			const vh = window.innerHeight;
			const t = Math.min(1, Math.max(0, (vh * 0.35 - r.top) / (r.height - vh * 0.5)));
			if (barRef.current) barRef.current.style.width = `${t * 100}%`;
			let on = -1;
			heads.forEach((h, i) => {
				if (h && h.getBoundingClientRect().top < vh * 0.3) on = i;
			});
			links.forEach((a, i) => a.classList.toggle("on", i === on));
		};
		const onScroll = () => {
			if (!tick) {
				tick = true;
				requestAnimationFrame(update);
			}
		};
		window.addEventListener("scroll", onScroll, { passive: true });
		window.addEventListener("resize", update);
		update();
		return () => {
			art.removeEventListener("click", onCopy);
			window.removeEventListener("scroll", onScroll);
			window.removeEventListener("resize", update);
		};
	}, [articleRef, barRef, tocRef]);
}

export default function Post({ postData, strip, older, newer, totals }) {
	const articleRef = useRef(null);
	const barRef = useRef(null);
	const tocRef = useRef(null);
	usePostEffects(articleRef, barRef, tocRef);

	const tags = postData.tags || [];
	const position = strip.ticks.findIndex((t) => t.me);

	return (
		<>
			<Head>
				<title>{`${postData.title} · Zachary Messinger`}</title>
				{postData.description && <meta name="description" content={postData.description} />}
				<meta property="og:title" content={postData.title} />
				<meta property="og:url" content={`${SITE_URL}/posts/${postData.id}`} />
				<link href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" rel="stylesheet" />
			</Head>
			<SwishJam>
				<div className="prog" aria-hidden="true">
					<b ref={barRef} />
				</div>
				<div className="wrap">
					<SiteHeader />
					<header className="head">
						<div className="kick caps">
							{tags.map((t, i) => (
								<span key={t} className={i === 0 ? "tag" : undefined}>
									{t}
								</span>
							))}
						</div>
						<h1>{postData.title}</h1>
						<div className="meta">
							<span>{fmt(postData.date, "MMM D YYYY")}</span>
							<span>&middot;</span>
							<span>{postData.readingMinutes} minute read</span>
						</div>
						<div
							className="where"
							role="img"
							aria-label={`This post on the timeline: number ${strip.ticks.length - position} of ${strip.ticks.length} posts`}
						>
							{strip.years.map((y) => (
								<span key={y.year} style={{ left: `${y.left}%` }}>
									{y.year}
								</span>
							))}
							{strip.ticks.map((t) => (
								<i key={t.id} className={t.me ? "me" : "pt"} style={{ left: `${t.left}%` }} />
							))}
						</div>
					</header>

					<div className="layout">
						{postData.toc.length > 1 && (
							<nav className="toc" aria-label="On this page" ref={tocRef}>
								<span className="caps">On this page</span>
								<ol>
									{postData.toc.map((h) => (
										<li key={h.id} className={h.depth > postData.toc[0].depth ? "sub" : undefined}>
											<a href={`#${h.id}`}>{h.text}</a>
										</li>
									))}
								</ol>
							</nav>
						)}
						<article ref={articleRef}>
							<div className="prose" dangerouslySetInnerHTML={{ __html: postData.contentHtml }} />
							<nav className="pn" aria-label="More writing">
								{older ? (
									<Link href={`/posts/${older.id}`}>
										<span className="caps">&larr; Previous</span>
										<span className="t">{older.title}</span>
										<span className="d">{fmt(older.date, "MMM YYYY")}</span>
									</Link>
								) : newer ? (
									<Link href={`/posts/${newer.id}`}>
										<span className="caps">Next &rarr;</span>
										<span className="t">{newer.title}</span>
										<span className="d">{fmt(newer.date, "MMM YYYY")}</span>
									</Link>
								) : (
									<span />
								)}
								<Link className="home" href="/">
									<span className="caps">Back to the line &rarr;</span>
									<span className="t">Everything since {strip.years[0].year}</span>
									<span className="d">
										{totals.posts} posts &middot; {totals.projects} projects
									</span>
								</Link>
							</nav>
						</article>
					</div>
					<SiteFooter />
				</div>
			</SwishJam>
		</>
	);
}

export async function getStaticProps({ params }) {
	const postData = await getPostData(params.id);
	const posts = getSortedPostsData(); // newest first
	const i = posts.findIndex((p) => p.id === params.id);
	const pick = (p) => (p ? { id: p.id, title: p.title, date: p.date } : null);

	return {
		props: {
			postData,
			strip: postStrip(posts, params.id),
			older: pick(posts[i + 1]),
			newer: pick(posts[i - 1]),
			totals: { posts: posts.length, projects: getSortedProjectsData().length + NOW.length },
		},
	};
}

export async function getStaticPaths() {
	return { paths: getAllPostIds(), fallback: false };
}

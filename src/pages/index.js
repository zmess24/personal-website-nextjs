import "../styles/master.scss";
import { useState } from "react";
import Header from "@/components/Header";
import Dropdown from "@/components/Dropdown";
import Footer from "@/components/Footer";
import Video from "@/components/Video";
import Head from "next/head";

export default function Home() {
	let [dropDownState, setDropDownState] = useState(false);
	let [dropDownData, setDropDownData] = useState({ name: "", data: [] });
	let projects = [];

	let handleToggle = (e) => {
		e.preventDefault();
		if (e.target.title === "open") {
			let data = e.target.textContent === "Projects" ? { name: "Projects", data: projects } : { name: "Blogs", data: [] };
			setDropDownData(data);
		}
		setDropDownState(!dropDownState);
	};
	return (
		<main id="container">
			<Head>
				<title>zacmessinger.com</title>
			</Head>
			<Video />
			<div className="wrapper">
				<Header dropDownState={dropDownState} handleClick={handleToggle} />
				{/* <Header handleClick={handleToggle} /> */}
				<Dropdown dropDownState={dropDownState} data={dropDownData} handleClick={handleToggle} />
				<section className="banner"></section>
				<Footer />
			</div>
		</main>
	);
}
